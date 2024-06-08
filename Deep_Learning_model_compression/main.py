import KnowledgeDistillation_LSTM
import KnowledgeDistillation_RNN


hidden_size_list = [16,32]
num_layers = 2

for i in range(1,1000):
    num_of_epoch = i

    for hidden_size in hidden_size_list:
        model_path = 'RNN\\model\\RNN_model_1.pt'
        KnowledgeDistillation_RNN.RNN_light_weight(model_path, hidden_size, num_layers, num_of_epoch)

        model_path = 'LSTM\\model\\LSTM_model_1.pt'
        KnowledgeDistillation_LSTM.LSTM_light_weight(model_path, hidden_size, num_layers, num_of_epoch)