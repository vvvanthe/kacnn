import torch
import torch.nn.functional as F
import math
from einops import rearrange

class KANConv2D(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=0, 
        dilatation=1, 
        groups=1,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        
    ):
        super(KANConv2D, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.kernel_size= kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.dilatation = dilatation


        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(kernel_size,kernel_size,in_channels, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)


        # grid2 = (
        #     (
        #         torch.arange(-spline_order, grid_size + spline_order + 1) * h
        #         + grid_range[0]
        #     )
        #     .expand(in_channels, -1)
        #     .contiguous()
        # )
        #self.register_buffer("grid2", grid2)


        self.base_weight = torch.nn.Parameter(torch.Tensor(out_channels, in_channels,kernel_size,kernel_size))

        self.convert_weight = torch.nn.Parameter(torch.Tensor(in_channels*(spline_order+grid_size), in_channels,1,1))
        
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_channels, in_channels,kernel_size,kernel_size, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_channels, in_channels,kernel_size,kernel_size)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        # n = self.in_channels*self.kernel_size*self.kernel_size
        
        # stdv = 1. / math.sqrt(n)
        # self.base_weight.data.uniform_(-stdv, stdv)
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        torch.nn.init.kaiming_uniform_(self.convert_weight, a=math.sqrt(5) * self.scale_base)

        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_channels, self.out_channels,self.kernel_size,self.kernel_size)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    rearrange(self.grid[:,:,:,self.spline_order : -self.spline_order],'k1 k2 c s -> s k1 k2 c') ,
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                #self.spline_scaler.data.uniform_(-stdv, stdv)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
    
        assert x.dim() == 4 and x.size(-1) == self.in_channels

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:,:,:, :-1]) & (x < grid[:,:,:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:,:,:, : -(k + 1)])
                / (grid[:,:,:, k:-1] - grid[:, :,:,: -(k + 1)])
                * bases[:, :, :,:,:-1]
            ) + (
                (grid[:,:,:, k + 1 :] - x)
                / (grid[:,:,:, k + 1 :] - grid[:,:,:, 1:(-k)])
                * bases[:,:,:, :, 1:]
            )
        assert bases.size() == (
            x.size(0),
            self.kernel_size,self.kernel_size,
            self.in_channels,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()
    
    # def b_splines_conv(self, x: torch.Tensor):
    #     """
    #     Compute the B-spline bases for the given input tensor.

    #     Args:
    #         x (torch.Tensor): Input tensor of shape (batch_size, in_channels).

    #     Returns:
    #         torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
    #     """
    #     # print('b_splines_conv x:',x.shape)


    #     assert x.dim() == 4 and x.size(1) == self.in_channels

    #     b, c, h, w = x.shape

    #     x= rearrange(x, 'b c h w -> (b h w) c')

    #     grid: torch.Tensor = (
    #         self.grid2
    #     )  # (in_features, grid_size + 2 * spline_order + 1)
    #     x = x.unsqueeze(-1)
    #     #print('b_spline grid ',grid.shape)
    #     bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
    #     for k in range(1, self.spline_order + 1):
    #         bases = (
    #             (x - grid[:, : -(k + 1)])
    #             / (grid[:, k:-1] - grid[:, : -(k + 1)])
    #             * bases[:, :, :-1]
    #         ) + (
    #             (grid[:, k + 1 :] - x)
    #             / (grid[:, k + 1 :] - grid[:, 1:(-k)])
    #             * bases[:, :, 1:]
    #         )
    #     bases= rearrange(bases, '(b h w) c s -> b (c s) h w',b=b,h=h, w=w)
        
    #     # assert bases.size() == (
    #     #     x.size(0),
    #     #     self.kernel_size,self.kernel_size,
    #     #     self.in_channels,
    #     #     self.grid_siz  e + self.spline_order,
    #     # )
    #     return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
  
        assert x.dim() == 4 and x.size(-1) == self.in_channels
        assert y.size() == (x.size(0), self.in_channels, self.out_channels,self.kernel_size,self.kernel_size)
        
        A = rearrange( self.b_splines(x),'a k1 k2 inc s -> inc k1 k2 a s') 
        B = rearrange(y,'a inc b k1 k2 -> inc k1 k2 a b') 
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = rearrange(solution,'inc k1 k2 s out-> out inc k1 k2 s')   # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_channels,
            self.in_channels,self.kernel_size,self.kernel_size,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.dim() == 4 and x.size(1) == self.in_channels
        input_base = self.base_activation(x)
        base_output = F.conv2d(input=input_base, weight=self.base_weight,bias=None, stride=self.stride, padding=self.padding,dilation=self.dilatation)
        x_convert = F.conv2d(x,self.convert_weight,padding=0,stride=1)  #F.conv2d(x,self.convert_weight,padding=0,stride=1)  self.b_splines_conv(x)
        spline_output = F.conv2d(
            x_convert,
            rearrange(self.scaled_spline_weight,'out inc k1 k2 s-> out (inc s) k1 k2'),bias=None, stride=self.stride, padding=self.padding,dilation=self.dilatation
        )
        return base_output + spline_output

 
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
if __name__ == '__main__':

    x = torch.ones((4, 3, 32, 32))
    layer = KANConv2D(3,64)
    conv = torch.nn.Conv2d(3,64,kernel_size=3)
    #layer.update_grid(x)
    y = layer(x)
    print(count_parameters(layer))
    
    
    print(count_parameters(conv))

    covn2 = KANConv2D(3, 64, 3, 1)

    y2 = covn2(x)
    print(y2.shape)
  