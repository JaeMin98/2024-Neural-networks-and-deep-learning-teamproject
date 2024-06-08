import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

torch.autograd.set_detect_anomaly(True)

# 장치 설정 (GPU가 가능하면 GPU를 사용하고, 아니면 CPU를 사용)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def RNN_light_weight(model_path, hidden_size, num_layers, num_of_epoch, num_of_epoch2=0):
    dataset_list = [
        ('dataset\\traindata\\train_var_mu.csv', num_of_epoch2),
        ('dataset\\testdata\\DLC_50kph_1.0_10ms.csv', num_of_epoch),
        ('dataset\\testdata\\DLC_tight_1.0_10ms.csv', num_of_epoch),
        ('dataset\\testdata\\test_0.4_10ms.csv', num_of_epoch),
        ('dataset\\testdata\\test_0.6_10ms.csv', num_of_epoch),
        ('dataset\\testdata\\test_0.85_10ms.csv', num_of_epoch)
    ]

    all_x_data = []
    all_y_data = []

    file_path = os.path.join("dataset", "traindata", "train_var_mu.csv")
    file_path = os.path.abspath(file_path)
    print("=== Train data file path ===", "\n", file_path)

    data = pd.read_csv(file_path, sep=",", header=0, dtype=float)
    x_data = torch.tensor(data.iloc[:, [0, 2, 3, 4, 5]].values, dtype=torch.float32).to(device)
    y_data = torch.tensor(data.iloc[:, 1:2].values, dtype=torch.float32).to(device)

    all_x_data.append(x_data)
    all_y_data.append(y_data)
            
    all_x_data = torch.cat(all_x_data, dim=0)
    all_y_data = torch.cat(all_y_data, dim=0)

    mean_x = torch.mean(all_x_data, dim=0)
    std_x = torch.std(all_x_data, dim=0)
    Max_Vy = max(abs(all_y_data))

    def csv_data_loader(file_path):
        print("=== Test data file path ===", "\n", file_path)

        test_data = pd.read_csv(file_path, sep=",", header=0, dtype=float)
        test_x_data = torch.tensor(test_data.iloc[:, [0, 2, 3, 4, 5]].values, dtype=torch.float32).to(device)
        test_y_data = torch.tensor(test_data.iloc[:, 1:2].values, dtype=torch.float32).to(device)
        test_x_data_normalized = (test_x_data - mean_x) / std_x
        
        return test_x_data, test_y_data, test_x_data_normalized

    class SmallRNN_Net(torch.nn.Module):
        def __init__(self):
            super(SmallRNN_Net, self).__init__()
            self.fc1 = torch.nn.Linear(6, hidden_size)
            self.fc2 = torch.nn.Linear(hidden_size, hidden_size)
            self.fc3 = torch.nn.Linear(hidden_size, 1)
            self.ln1 = torch.nn.LayerNorm(hidden_size)
            self.ln2 = torch.nn.LayerNorm(hidden_size)
            self.dropout = torch.nn.Dropout()
            self.gelu = torch.nn.GELU()

        def forward(self, x):
            x = self.gelu(self.ln1(self.fc1(x)))
            x = self.dropout(x)
            x = self.gelu(self.ln2(self.fc2(x)))
            x = self.dropout(x)
            x = torch.tanh(self.fc3(x))
            return x

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

    student_model = SmallRNN_Net().to(device)

    teacher_model = torch.jit.load(model_path)
    teacher_model.to(device)
    teacher_model.eval()

    optimizer = optim.Adam(student_model.parameters(), lr=0.001)
    temperature = 2.0
    criterion = DistillationLoss(temperature)
    prev_predictions_T = torch.zeros(1, 1).to(device)
    prev_predictions_S = torch.zeros(1, 1).to(device)

    for dataset_path, num_epochs in dataset_list:
        test_x_data, test_y_data, test_x_data_normalized = csv_data_loader(dataset_path)

        for epoch in range(num_epochs):
            student_model.train()
            running_loss = 0.0
            for i in range(len(test_x_data)):
                inputs = test_x_data_normalized[i].unsqueeze(0)
                
                optimizer.zero_grad()

                with torch.no_grad():
                    current_input_T = torch.cat([inputs, prev_predictions_T.detach()], dim=1)
                    teacher_outputs = teacher_model(current_input_T)
                    prev_predictions_T = teacher_outputs

                current_input_S = torch.cat([inputs, prev_predictions_S.detach()], dim=1)
                student_outputs = student_model(current_input_S)
                prev_predictions_S = student_outputs

                loss = criterion(student_outputs, teacher_outputs, test_y_data[i])
                loss.backward(retain_graph=True)
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}, Loss: {running_loss / len(test_x_data)}")

    scripted_model = torch.jit.script(student_model)
    torch.jit.save(scripted_model, f'RNN_student_model_{hidden_size}_{num_layers}_{num_of_epoch}.pt')

    print("Student model training complete and saved")

# 모델 학습 실행 예시
# RNN_light_weight('path_to_teacher_model.pt', hidden_size=128, num_layers=2, num_of_epoch=10)
