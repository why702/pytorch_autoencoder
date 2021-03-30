import osimport sysimport cv2import matplotlib.pyplot as pltimport numpy as npimport progressbarimport torchimport torch.nn.functional as Ffrom torch import nnfrom torch.autograd import Variablefrom torch.utils.tensorboard import SummaryWritersys.path.append("../pytorch_autoencoder")from utils.data_augmentation_fp import PerfDatasetfrom utils.combine_genuines_fpdbindex import parse_indexfrom ae_model import autoencoderfrom ae_perf_loss import output_ae_imgsdef align_img(img, theta, device):    grid = F.affine_grid(theta[:, :2], img.size()).to(device=device)    tans_img = F.grid_sample(img, grid)    return tans_imgdef mask_img(img, theta, device):    mask = torch.tensor((), dtype=torch.float32)    mask = mask.new_ones(img.shape).to(device=device)    grid = F.affine_grid(theta[:, :2], mask.size()).to(device=device)    mask = F.grid_sample(mask, grid)    img = img * mask    return imgdef Perf_MSE(template, verify, theta, device):    # print(theta[0])    trans_template = align_img(template, theta, device)    mask_verify = mask_img(verify, theta, device)    loss = nn.MSELoss()(trans_template, mask_verify)    # for i in range(10):    #     print(theta[i])    #     t0 = template.data.cpu().numpy().astype('uint8')[i][0]    #     cv2.imshow("t0", t0)    #     v0 = verify.data.cpu().numpy().astype('uint8')[i][0]    #     cv2.imshow("v0", v0)    #     t = trans_template.data.cpu().numpy().astype('uint8')[i][0]    #     cv2.imshow("t", t)    #     v = mask_verify.data.cpu().numpy().astype('uint8')[i][0]    #     cv2.imshow("v", v)    #     merge = np.zeros((t0.shape[0],t0.shape[1],3)).astype('uint8')    #     merge[:,:,1] = t * 0.5    #     merge[:,:,2] = v * 0.5    #     cv2.imshow("m", merge)    #     cv2.waitKey(0)    return lossif __name__ == '__main__':    num_epochs = 100000    batch_size = 64    # learning_rate = 1e-3    learning_rate = 1e-5    img_width = 200    img_height = 200    model_width = 224    model_height = 224    pad_width = int((model_width - img_width) / 2)    pad_height = int((model_height - img_height) / 2)    np_pad = ((pad_height, model_height - img_height - pad_height), (pad_width, model_width - img_width - pad_width))    use_cuda = torch.cuda.is_available()    device = torch.device('cuda:0' if use_cuda else 'cpu')    if not os.path.exists('./dc_img'):        os.mkdir('./dc_img')    if not os.path.exists('./model'):        os.mkdir('./model')    # get perf pair and result    gen_file = "D:\\PB_bat\\2021_0210_R3063\\dec_egistec_200x200_cardo_CH1AJA_100K_R120_t2560000_dry94_L0\\test\\genuines.txt"    index_file = "D:\\git\\Dev2930_NP\\image_bin\\i.fpdbindex"    output_file = 'pair_info.csv'    # get dataset    dataset = PerfDataset(gen_file=gen_file, index_file=index_file, csv_file=output_file, img_width=img_width,                          img_height=img_height, pad_width=np_pad, RBS=True, TRANS=True, PI=True)    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)    fpdbindex_data = parse_index(index_file)    writer = SummaryWriter()    model = autoencoder().to(device=device)    criterion = nn.MSELoss()    weighting_file = './conv_autoencoder_1900.pth'    if os.path.exists(weighting_file):        model.load_state_dict(torch.load(weighting_file))        model.eval()        model.to(device)    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,                                 weight_decay=1e-5)    # and a learning rate scheduler    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,                                                   step_size=1000,                                                   gamma=0.5)    N_TEST_IMG = 5    f, a = plt.subplots(3, N_TEST_IMG, figsize=(10, 6))    plt.ion()  # continuously plot    for epoch in range(num_epochs):        train_MSE_loss = 0        train_Perf_MSE_loss = 0        total_step = int(dataset.size / batch_size)        with progressbar.ProgressBar(max_value=total_step) as bar:            for step, (enroll_raw, enroll_ipp, verify_raw, verify_ipp, _, theta) in enumerate(dataloader):                img_e = Variable(enroll_raw).to(device=device)                img_v = Variable(verify_raw).to(device=device)                label_e = Variable(enroll_ipp).to(device=device)                label_v = Variable(verify_ipp).to(device=device)                # ===================forward=====================                output_e = model(img_e)                output_v = model(img_v)                output_e = (output_e - torch.min(output_e)) / (torch.max(output_e) - torch.min(output_e)) * 255                output_v = (output_v - torch.min(output_v)) / (torch.max(output_v) - torch.min(output_v)) * 255                output_t = torch.cat((output_e, output_v), 0)                label_t = torch.cat((label_e, label_v), 0)                loss_t = criterion(output_t, label_t)                train_MSE_loss += loss_t.data                output_e_ = dataset.crop_img(output_e)                output_v_ = dataset.crop_img(output_v)                perf_mse_loss = Perf_MSE(output_e_, output_v_, theta, device)                train_Perf_MSE_loss += perf_mse_loss.data                # ===================backward====================                optimizer.zero_grad()                # loss_t.backward()                loss_t.backward(retain_graph=True)                perf_mse_loss.backward()                optimizer.step()                # print([x.grad for x in model.parameters()])                if step == 0:                    # show training result                    if epoch % 10 == 0:                        # plotting decoded image                        for i in range(N_TEST_IMG):                            a0 = np.reshape(img_v.data.cpu().numpy()[i], (model_width, model_height))                            a0 = a0[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                            a1 = np.reshape(label_v.data.cpu().numpy()[i], (model_width, model_height))                            a1 = a1[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                            a2 = np.reshape(output_v.data.cpu().numpy()[i], (model_width, model_height))                            a2 = a2[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                            a[0][i].clear()                            a[0][i].imshow(a0, cmap='gray')                            a[1][i].clear()                            a[1][i].imshow(a1, cmap='gray')                            a[2][i].clear()                            a[2][i].imshow(a2, cmap='gray')                            pass                        plt.draw()                        # plt.pause(0.05)                        plt.savefig('./dc_img/image_{}.png'.format(epoch))                        pass                bar.update(step)            pass  # 1 train epoch end        writer.add_scalar('Loss/train', train_MSE_loss, epoch)        writer.add_scalar('Loss/train_perf', train_Perf_MSE_loss, epoch)        # test        if epoch % 50 == 0:            output_ae_imgs(index_file, fpdbindex_data, dataset, model, device)        lr_scheduler.step()        # ===================log========================        print('epoch [{}/{}], loss:{:.4f}, perf_loss:{:.4f}, lr={}'              .format(epoch + 1, num_epochs, train_MSE_loss, train_Perf_MSE_loss, optimizer.param_groups[0]['lr']))        if epoch % 100 == 0 and epoch > 0:            torch.save(model.state_dict(), './model/conv_autoencoder_{}.pth'.format(epoch))            print('save ./model/conv_autoencoder_{}.pth'.format(epoch))    writer.close()