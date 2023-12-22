from typing import Tuple

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange


# Macros.

def Normalization(input_channels: int) -> nn.Module:

    return nn.GroupNorm(
        num_groups=32,
        num_channels=input_channels,
    )


def Convolution(input_channels: int, output_channels: int) -> nn.Module:

    return nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    )


def Downsample(input_channels: int, output_channels: int) -> nn.Module:

    return nn.Conv2d(
        in_channels=input_channels,
        out_channels=output_channels,
        kernel_size=3,
        stride=2,
        padding=1,
    )


def Chain(module: nn.Module, arguments: Tuple[int, ...]) -> nn.Module:

    return nn.Sequential(*[
        module(
            arguments[i], 
            arguments[i + 1],
        ) for i in range(len(arguments) - 1)
    ])


# Modules.

class DownsampleBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.normalization = Normalization(input_channels=input_channels)

        self.downsample = Downsample(
            input_channels=input_channels,
            output_channels=output_channels,
        ) 

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.normalization(x)
        x = self.downsample(x)
        x = F.leaky_relu(x)

        return x
        

class ResidualBlock(nn.Module):

    def __init__(self, input_channels: int) -> None:
        super().__init__()

        self.normalization = Normalization(input_channels=input_channels)

        self.convolution = Convolution(
            input_channels=input_channels,
            output_channels=input_channels,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        z = self.normalization(x)
        z = self.convolution(z)
        z = F.leaky_relu(z)

        return x + z


class EncoderBlock(nn.Module):

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()

        self.residual_block_1 = ResidualBlock(input_channels=input_channels)
        self.residual_block_2 = ResidualBlock(input_channels=input_channels)

        self.downsample_block = DownsampleBlock(
            input_channels=input_channels,
            output_channels=output_channels,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.residual_block_1(x)
        x = self.residual_block_2(x)
        x = self.downsample_block(x)

        return x


class Encoder(nn.Module):
    
    def __init__(
        self, 
        input_channels: int, 
        hidden_channels: Tuple[int, ...],
        embedding_channels: int,
    ) -> None:
        super().__init__()

        self.convolution_1 = Convolution(
            input_channels=input_channels,
            output_channels=hidden_channels[0],
        )

        self.convolution_2 = Convolution(
            input_channels=hidden_channels[-1],
            output_channels=embedding_channels,
        )

        self.encoder_blocks = Chain(
            module=EncoderBlock,
            arguments=hidden_channels,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        batch_size = x.size(0)

        x = rearrange(x, 'b a c h w -> (b a) c h w')
        x = self.convolution_1(x)
        x = self.encoder_blocks(x)
        x = self.convolution_2(x)
        x = F.adaptive_avg_pool2d(x, output_size=1)
        x = rearrange(x, '(b a) c () () -> b a c', b=batch_size)

        return x 


class Attention(nn.Module):
    
    def __init__(self, embedding_channels: int, heads: int) -> None:
        super().__init__()

        self.heads = heads

        self.linear_1 = nn.Linear(
            in_features=embedding_channels, 
            out_features=embedding_channels * 3, 
            bias=False,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_channels, 
            out_features=embedding_channels, 
            bias=False,
        )
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        x = self.linear_1(x)
        q, k, v = rearrange(x, 'b s (k h e) -> k b h s e', k=3, h=self.heads)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        x = self.linear_2(rearrange(x, 'b h s e -> b s (h e)'))
      
        return x


class MLP(nn.Module):

    def __init__(self, embedding_channels: int) -> None:
        super().__init__()

        self.linear_1 = nn.Linear(
            in_features=embedding_channels, 
            out_features=embedding_channels * 4,
        )

        self.linear_2 = nn.Linear(
            in_features=embedding_channels * 4,
            out_features=embedding_channels,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.linear_1(x)
        x = F.gelu(x)
        x = self.linear_2(x)

        return x


class DecoderBlock(nn.Module):

    def __init__(self, embedding_channels: int, heads: int) -> None:
        super().__init__()

        self.attention = Attention(
            embedding_channels=embedding_channels,
            heads=heads,
        )

        self.mlp = MLP(embedding_channels=embedding_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x + self.attention(x, None)
        x = x + self.mlp(x)

        return x


class Decoder(nn.Module):

    def __init__(
        self,
        embedding_channels: int,
        vocabulary_size: int,
        heads: int,
        blocks: int,
    ) -> None:
        super().__init__()

        self.linear = nn.Linear(
            in_features=embedding_channels,
            out_features=vocabulary_size,
        )

        self.decoder_blocks = Chain(
            module=lambda _, __: DecoderBlock(
                embedding_channels=embedding_channels,
                heads=heads,
            ),
            arguments=(1,) * (heads + 1),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.decoder_blocks(x)
        x = self.linear(x)
        x = F.log_softmax(x, dim=-1)

        return x
    

@dataclass(frozen=True)
class Uniformity1Configuration:
    vocabulary_size: int
    embedding_dimension: int
    heads: int
    blocks: int
    input_channels: int
    encoder_channels: Tuple[int, ...]


class Uniformity1(nn.Module):

    def __init__(self, configuration: Uniformity1Configuration) -> None:
        super().__init__()

        self.encoder = Encoder(
            input_channels=configuration.input_channels,
            hidden_channels=configuration.encoder_channels,
            embedding_channels=configuration.embedding_dimension,
        )

        self.decoder = Decoder(
            embedding_channels=configuration.embedding_dimension,
            vocabulary_size=configuration.vocabulary_size,
            heads=configuration.heads,
            blocks=configuration.blocks,
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.encoder(x)
        x = self.decoder(x)

        return x
