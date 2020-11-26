import os
import pickle
import torch


class FileUtil():
    '''
    ディレクトリの作成など
    '''

    def __init__():
        pass

    @staticmethod
    def mkdir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def save_checkpoint(save_path, model, optimizer, epoch):
        # save
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        },
            save_path
        )
        return 0

    @staticmethod
    def load_checkpoint(path, model, optimizer):
        # load
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']

        return model, optimizer, epoch

    @classmethod
    def image_save(self, image_dict, output_name='./output/temp/image/converted.pkl'):
        """
        推論した画像をpickle形式で保存する
        """
        # フォルダ作成
        self.mkdir(os.path.split(output_name)[0])
        # pickleで保存
        with open(output_name, 'wb') as f:
            pickle.dump(image_dict, f)

    @staticmethod
    def load_image(file_path):
        """
        pickleした画像を再度読み込む
        """

        with open(file_path, 'rb') as f:
            image_dict = pickle.load(f)
        return image_dict

    @staticmethod
    def list_check(input):
        """
        リストでないものをリストに変換
        リストはそのまま返す
        """
        if type(input) is list:
            return input
        else:
            return [input]
