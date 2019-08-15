import torch


def loadModel(model, model_path=None):
        if model_path != '':
            if torch.cuda.is_available():
                pretrained_dict = torch.load(model_path)
            else:
                pretrained_dict = torch.load(model_path,map_location=torch.device('cpu'))
            model_dict = model.state_dict()
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            for k in pretrained_dict:
                print('key:',k)
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            print('Load model:{}'.format(model_path))
        return model