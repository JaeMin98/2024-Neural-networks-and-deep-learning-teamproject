import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset



def LSTM_light_weight(model_path, hidden_size, num_layers, num_of_epoch):
    # 데이터 파일 경로 설정
    dataset_list = [('dataset\\traindata\\train_var_mu.csv',num_of_epoch),
                    ('dataset\\testdata\\DLC_50kph_1.0_10ms.csv',num_of_epoch),
                    ('dataset\\testdata\\DLC_tight_1.0_10ms.csv',num_of_epoch),
                    ('dataset\\testdata\\test_0.4_10ms.csv',num_of_epoch),
                    ('dataset\\testdata\\test_0.6_10ms.csv',num_of_epoch),
                    ('dataset\\testdata\\test_0.85_10ms.csv',num_of_epoch)
                    ]
    
    dataset_list = [('dataset\\traindata\\train_var_mu.csv',0),
                    ('dataset\\testdata\\DLC_50kph_1.0_10ms.csv',1),
                    ('dataset\\testdata\\DLC_tight_1.0_10ms.csv',1),
                    ('dataset\\testdata\\test_0.4_10ms.csv',0),
                    ('dataset\\testdata\\test_0.6_10ms.csv',0),
                    ('dataset\\testdata\\test_0.85_10ms.csv',0)
                    ]
    
    all_x_data = []
    all_y_data = []

    file_path = os.path.join("dataset","traindata","train_var_mu.csv")
    file_path = os.path.abspath(file_path)
    print("===trin data file path===", "\n", file_path)

    # Pandas를 이용해 파일 읽기
    data = pd.read_csv(file_path, sep=",", header=0, dtype=float)
    x_data = torch.tensor(data.iloc[:, [0,2,3,4,5]].values, dtype=torch.float32)
    y_data = torch.tensor(data.iloc[:, 1:2].values, dtype=torch.float32)

    # 전체 데이터 리스트에 추가
    all_x_data.append(x_data)
    all_y_data.append(y_data)
            
    # 리스트에 저장된 모든 데이터를 하나의 텐서로 결합
    all_x_data = torch.cat(all_x_data, dim=0)
    all_y_data = torch.cat(all_y_data, dim=0)

    # 전체 데이터에 대한 평균과 분산 계산
    mean_x = torch.mean(all_x_data, dim=0)
    std_x = torch.std(all_x_data, dim=0)

    Max_Vy = max(abs(all_y_data))

    def csv_data_loader(file_path):
        print("===train data file path===", "\n", file_path)

        test_data = pd.read_csv(file_path, sep=",", header=0, dtype=float)
        test_x_data = torch.tensor(test_data.iloc[:, [0,2,3,4,5]].values, dtype=torch.float32)
        test_y_data = torch.tensor(test_data.iloc[:, 1:2].values, dtype=torch.float32)
        test_x_data_normalized = (test_x_data - mean_x) / std_x  # mean과 std는 훈련 데이터셋에서 계산된 값 사용
        
        return test_x_data, test_y_data, test_x_data_normalized

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
            self.hidden_size = hidden_size  # hidden size 축소
            self.num_layers = num_layers    # layer 수 축소
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
            distillation_loss = nn.MSELoss()(student_outputs, teacher_outputs)
            targets = targets.view_as(student_outputs)
            student_loss = nn.MSELoss()(student_outputs, targets)
            return distillation_loss + student_loss

    teacher_model = LSTMNet(5, 64, 2, 1)

    # 모델 초기화
    input_size = 5
    output_size = 1
    student_model = SmallLSTMNet(input_size, hidden_size, num_layers, output_size)

    # Teacher 모델 로드 (path_to_your_teacher_model.pth를 실제 파일 경로로 변경)
    model_data = torch.jit.load(model_path)

    teacher_model.load_state_dict(model_data.state_dict())
    teacher_model.eval()

    # Student 모델 학습 설정
    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    temperature = 2.0
    criterion = DistillationLoss(temperature)


    for dataset_path, num_epochs in dataset_list:
        test_x_data, test_y_data, test_x_data_normalized = csv_data_loader(dataset_path)

        for epoch in range(num_epochs):
            student_model.train()
            running_loss = 0.0
            for i in range(len(test_x_data)):
                inputs = test_x_data_normalized[i].unsqueeze(0).unsqueeze(0)
                optimizer.zero_grad()

                with torch.no_grad():
                    teacher_outputs = teacher_model(inputs)
                
                student_outputs = student_model(inputs)
                loss = criterion(student_outputs, teacher_outputs, test_y_data[i])
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(test_x_data)}")


    # 모델을 TorchScript로 변환
    scripted_model = torch.jit.script(student_model)
    # TorchScript 모델 저장
    torch.jit.save(scripted_model, 'LSTM_student_model_{0}_{1}_{2}.pt'.format(hidden_size,num_layers,num_of_epoch))

    print("Student model training complete and saved")

num_of_epoch = 0

hidden_size = 32
num_layers = 2
model_path = 'LSTM\\model\\LSTM_model_1.pt'
LSTM_light_weight(model_path, hidden_size, num_layers, num_of_epoch)


hidden_size = 16
num_layers = 2
model_path = 'LSTM\\model\\LSTM_model_1.pt'
LSTM_light_weight(model_path, hidden_size, num_layers, num_of_epoch)