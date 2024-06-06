import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torch.utils.data import DataLoader, TensorDataset

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

# device 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# teacher 모델 로드
teacher_model_path = "LSTM/model/LSTM_model_1.pt"
teacher_model = torch.jit.load(teacher_model_path).to(device)
teacher_model.eval()

# student 모델 정의 (더 작은 hidden_size 사용)
student_model = LSTMNet(input_size=5, hidden_size=32, num_layers=2, output_size=1).to(device)

# Distillation Loss 정의
class DistillationLoss(nn.Module):
    def __init__(self, student, teacher, temperature=2.0, alpha=0.3):
        super(DistillationLoss, self).__init__()
        self.student = student
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.MSELoss()

    def forward(self, x, y):
        student_logits = self.student(x)
        with torch.no_grad():
            teacher_logits = self.teacher(x)
        loss = self.criterion(student_logits, y)
        distillation_loss = nn.KLDivLoss()(F.log_softmax(student_logits / self.temperature, dim=1),
                                           F.softmax(teacher_logits / self.temperature, dim=1)) * (self.temperature ** 2)
        return loss * (1 - self.alpha) + distillation_loss * self.alpha

# 데이터셋 준비 (예시 데이터 사용)
x_data = torch.randn(100, 10, 5)  # (batch_size, sequence_length, input_size)
y_data = torch.randn(100, 1)      # (batch_size, output_size)
dataset = TensorDataset(x_data, y_data)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# 옵티마이저 정의
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# 학습 루프
num_epochs = 10000
criterion = DistillationLoss(student_model, teacher_model)

for epoch in range(num_epochs):
    student_model.train()
    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        loss = criterion(inputs, targets)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")
    if(loss.item() < 0.0000001): break

# Pruning 적용 (원하는 대로)
for name, module in student_model.named_modules():
    if isinstance(module, nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.4)
        prune.remove(module, 'weight')

# Pruned 모델 저장
pruned_scripted_student_model = torch.jit.script(student_model)
pruned_student_model_path = "LSTM/model/LSTM_model_1_pruned_student.pt"
torch.jit.save(pruned_scripted_student_model, pruned_student_model_path)

print(f'Model saved at {pruned_student_model_path}')
