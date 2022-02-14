import torch
import torch.nn as nn


class Block(nn.Module):
    
    def __init__(self,
                 input_size,
                 output_size,
                 use_batch_norm=True,
                 dropout_p=.4):
        self.input_size = input_size
        self.output_size = output_size
        self.use_batch_norm = use_batch_norm
        self.dropout_p = dropout_p
        
        super().__init__()
        
        def get_regularizer(use_batch_norm, size):
            return nn.BatchNorm1d(size) if use_batch_norm else nn.Dropout(dropout_p)
        # 현재 우리가 만드는 중인 모델 아키텍처에서는 Linear 레이어 다음에 BatchNorm을 넣기 때문에, Linear Layer의 출력 사이즈가 BatchNorm의 입력 사이즈가 된다.
        # 즉, size = Linear Layer의 출력 사이즈 즉, = output_size
        
        self.block = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.LeakyReLU(),
            get_regularizer(use_batch_norm, output_size), #위에서 말한 바와 같이 output_size가 nn.BatchNomr1d() 함수의 입력 인자로 들어감
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)
        y = self.block(x)
        # |y| = (batch_size, output_size)
        
        return y

    
class ImageClassifier(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 hidden_sizes=[500, 400, 300, 200, 100], # Tuning을 하기 위해 은닉층 사이즈를 이렇게 입력으로 받을 수 있게 함!!!
                 use_batch_norm=True,
                 dropout_p=.3):
        
        super().__init__()

        assert len(hidden_sizes) > 0, "You need to specify hidden layers"  # 예외 처리한 거. "Hidden size가 없으면 안 된다"는 의미

        last_hidden_size = input_size
        blocks = []
        for hidden_size in hidden_sizes:
            blocks += [Block(
                last_hidden_size,
                hidden_size,
                use_batch_norm,
                dropout_p
            )]
            last_hidden_size = hidden_size
        
        self.layers = nn.Sequential(
            *blocks,
            nn.Linear(last_hidden_size, output_size),
            nn.LogSoftmax(dim=-1),
        )
        
    def forward(self, x):
        # |x| = (batch_size, input_size)        
        y = self.layers(x)
        # |y| = (batch_size, output_size)
        
        return y
