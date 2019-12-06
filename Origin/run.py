import torch
import torch.nn as nn
import torch.optim as optim
from model import get_style_model_and_loss


def get_input_param_optimier(input_img):

    input_param = nn.Parameter(input_img.data)
    optimizer = optim.LBFGS([input_param])
    return input_param, optimizer


def run_style_transfer(content_img, style_img, input_img, num_epoches=300, alpha=100000, beta=1):

    model, style_loss_list, content_loss_list = get_style_model_and_loss(style_img, content_img)
    input_param, optimizer = get_input_param_optimier(input_img)
    epoch = [0]
    
  
    
    def closure():    
        input_param.data.clamp_(0, 1)

        model(input_param)
        style_score = 0
        content_score = 0
        optimizer.zero_grad()

        for sl in style_loss_list: 
            style_score = style_score + sl.loss

        for cl in content_loss_list:
            content_score = content_score + cl.loss

        style_score = alpha * style_score
        content_score = beta * content_score
        loss = style_score + content_score
        loss.backward()

        epoch [0] += 1
        if epoch[0] % 50 == 0:
            print('run {}'.format(epoch [0]))
            print('Style Loss: {:.10f} Content Loss: {:.10f}'.format(
                style_score.data, content_score.data))
            print()

        return   style_score + content_score

    while epoch [0] < num_epoches:           
        optimizer.step(closure)        
        input_param.data.clamp_(0, 1)

    return input_param.data
