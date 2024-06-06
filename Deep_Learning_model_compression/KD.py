import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 데이터 파일 경로 설정
file_path = 'dataset\\traindata\\drive_data_train.csv'
print("===train data file path===", "\n", file_path)

# Pandas를 이용해 파일 읽기
data = pd.read_csv(file_path, sep=",", header=0, dtype=float)
x_data = torch.tensor(data.iloc[:, [0, 2, 3, 4, 5]].values, dtype=torch.float32)
y_data = torch.tensor(data.iloc[:, 1:2].values, dtype=torch.float32)

# 전체 데이터 리스트에 추가
all_x_data = [x_data]
all_y_data = [y_data]

# 리스트에 저장된 모든 데이터를 하나의 텐서로 결합
all_x_data = torch.cat(all_x_data, dim=0)
all_y_data = torch.cat(all_y_data, dim=0)

# 데이터셋 및 데이터로더 생성
dataset = TensorDataset(all_x_data, all_y_data)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 기존 Teacher LSTMNet 모델 정의
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

# Student LSTMNet 모델 정의 (경량화된 버전)
class SmallLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(SmallLSTMNet, self).__init__()
        self.hidden_size = hidden_size // 2  # hidden size 축소
        self.num_layers = num_layers // 2    # layer 수 축소
        self.lstm = nn.LSTM(input_size, self.hidden_size, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_size, output_size)
        self.tanh = nn.Tanh()

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        out = self.tanh(out)
        return out

# Distillation Loss 함수 정의
class DistillationLoss(nn.Module):
    def __init__(self, temperature):
        super(DistillationLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_outputs, teacher_outputs, targets):
        soft_targets = torch.softmax(teacher_outputs / self.temperature, dim=1)
        soft_student_outputs = torch.log_softmax(student_outputs / self.temperature, dim=1)
        distillation_loss = self.criterion(soft_student_outputs, soft_targets) * (self.temperature ** 2)
        student_loss = nn.MSELoss()(student_outputs, targets)
        return distillation_loss + student_loss

# 모델 초기화
input_size = 5
hidden_size = 20
num_layers = 2
output_size = 1

teacher_model = LSTMNet(5, 64, 2, 1)
student_model = SmallLSTMNet(input_size, hidden_size, num_layers, output_size)

# Teacher 모델 로드 (path_to_your_teacher_model.pth를 실제 파일 경로로 변경)
model_path = 'LSTM\\model\\LSTM_model_1.pt'
model_data = torch.load(model_path)

teacher_model.load_state_dict(model_data.state_dict())
teacher_model.eval()

# Student 모델 학습 설정
optimizer = optim.Adam(student_model.parameters(), lr=0.001)
temperature = 5.0
criterion = DistillationLoss(temperature)

num_epochs = 10

for epoch in range(num_epochs):
    student_model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.unsqueeze(1)  # 차원 추가
        optimizer.zero_grad()

        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)
        
        student_outputs = student_model(inputs)
        loss = criterion(student_outputs, teacher_outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}")

# Student 모델 저장
torch.save(student_model.state_dict(), 'student_model.pth')
print("Student model training complete and saved as 'student_model.pth'")
