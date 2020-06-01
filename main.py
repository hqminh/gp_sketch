from experiments import *


if __name__ == '__main__':
    prefix = './29MayExp7/'
    result_folder = './1JunExpSet1/'
    if not os.path.isdir(result_folder):
        os.mkdir(result_folder)
    # dataset = ['abalone', 'diabetes']
    # method = ['full', 'ssgp_32', 'ssgp_64', 'ssgp_128', 'ssgp_256', 'ssgp_512', 'ssgp_1024']
    # seed = [1001, 1002, 1003, 1004, 1005]
    # vae = ['abalone.pth', 'diabetes.pth']
    vae = ['encoder_30.pth']
    dataset = ['abalone']
    method = ['ssgp_32', 'full']
    seed = [1001]

    for i, ds in enumerate(dataset):
        vae_model = torch.load(prefix + vae[i])
        for j, mt in enumerate(method):
            exp1 = Experiment(dataset=ds, method=mt, embedding=True, vae_model=vae_model)
            exp2 = Experiment(dataset=ds, method=mt, embedding=False, vae_model=vae_model)
            for s in seed:
                exp1.deploy(seed=s, savefile=result_folder + '_'.join([ds, mt, str(s)]) + '.txt')
                exp2.deploy(seed=s, savefile=result_folder + '_'.join([ds, mt, str(s)]) + '.txt')