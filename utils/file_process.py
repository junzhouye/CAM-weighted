import os


def get_file_name(root):
    # 获得一个文件夹下的所有文件名
    file_list = os.listdir(root)
    return file_list


def make_file(file_name):
    if not os.path.exists(file_name):
        os.makedirs(file_name)


if __name__ == "__main__":
    print(get_file_name("../pth"))

    a = ['adversarial_model_epoch10.pth',
         'adversarial_model_epoch20.pth',
         'adversarial_model_epoch30.pth',
         'adversarial_model_epoch40.pth',
         'adversarial_model_epoch50.pth',
         'cam_weighted_adversarial_model_mode1_epoch10.pth',
         'cam_weighted_adversarial_model_mode1_epoch20.pth',
         'cam_weighted_adversarial_model_mode1_epoch30.pth',
         'cam_weighted_adversarial_model_mode1_epoch40.pth',
         'cam_weighted_adversarial_model_mode1_epoch50.pth',
         'cam_weighted_trade_model_mode_1_beta_6_epoch10.pth',
         'cam_weighted_trade_model_mode_1_beta_6_epoch20.pth',
         'cam_weighted_trade_model_mode_1_beta_6_epoch30.pth',
         'cam_weighted_trade_model_mode_1_beta_6_epoch40.pth',
         'cam_weighted_trade_model_mode_1_beta_6_epoch50.pth',
         'standard_model-epoch10.pth',
         'standard_model-epoch20.pth',
         'standard_model-epoch30.pth',
         'standard_model-epoch40.pth',
         'standard_model-epoch50.pth',
         'trade_model_beta_6-epoch10.pth',
         'trade_model_beta_6-epoch20.pth',
         'trade_model_beta_6-epoch30.pth',
         'trade_model_beta_6-epoch40.pth',
         'trade_model_beta_6-epoch50.pth'
         ]
