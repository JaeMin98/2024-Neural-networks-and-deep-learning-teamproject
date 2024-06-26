import KnowledgeDistillation_LSTM
import KnowledgeDistillation_RNN


hidden_size_list = [16,32]
num_layers = 2
num_of_epoch2 = 0

for i in range(1,2):
    num_of_epoch = i
    num_of_epoch2 = 0

    if(num_of_epoch2 >3) : num_of_epoch2 = 1

    for hidden_size in hidden_size_list:
        model_path = 'RNN\\model\\RNN_64_64_model_1.pt'
        KnowledgeDistillation_RNN.RNN_light_weight(model_path, hidden_size, num_layers, num_of_epoch, num_of_epoch2)

        # model_path = 'LSTM\\model\\LSTM_model_1.pt'
        # KnowledgeDistillation_LSTM.LSTM_light_weight(model_path, hidden_size, num_layers, num_of_epoch, num_of_epoch2)