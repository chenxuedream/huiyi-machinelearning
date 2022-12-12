import torch
import torch.nn
import onnx

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = torch.load('./ImageToEmotionclassifier.pt')
with torch.no_grad():
    model.eval()
    input_names = ['input']
    output_names = ['output']
    x = torch.randn(1,3,256,256,requires_grad=True).to(device)
    torch.onnx.export(model, x, 'ImageToEmotionclassifier.onnx', input_names=input_names, output_names=output_names, verbose='True')
