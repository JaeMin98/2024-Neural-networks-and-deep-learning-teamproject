import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# LSTMNet 모델 정의
class LSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMNet, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # 마지막 시점의 출력만 사용
        out = self.tanh(out)  # tanh 활성화 함수 적용
        return out

# 모델 초기화
input_size = 5
hidden_size = 64
num_layers = 2
output_size = 1
model = LSTMNet(input_size, hidden_size, num_layers, output_size)

# 스크립트된 모델 로드
model_path = "LSTM/model/LSTM_model_1.pt"
scripted_model = torch.jit.load(model_path)

# 스크립트된 모델의 state_dict를 PyTorch 모델로 복사
model.load_state_dict(scripted_model.state_dict())

# Pruning 적용 전 파라미터 개수 출력
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_zero_parameters(model):
    return sum(torch.sum(p == 0).item() for p in model.parameters() if p.requires_grad)

print(f'Original model parameter count: {count_parameters(model)}')

# Pruning 적용
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=1.0)
        prune.remove(module, 'weight')  # pruning mask를 제거하여 가중치 업데이트

# Pruning 후 파라미터 개수 출력
print(f'Pruned model parameter count: {count_parameters(model)}')
print(f'Number of zero parameters after pruning: {count_zero_parameters(model)}')

# 모델 스크립팅 및 저장 (Pruning된 모델)
pruned_scripted_model = torch.jit.script(model)
pruned_model_path = "LSTM/model/LSTM_model_1_pruned.pt"
torch.jit.save(pruned_scripted_model, pruned_model_path)
