import osimport sysimport matplotlib.pyplot as pltimport numpy as npimport progressbarimport torchfrom torch import nnfrom torch.autograd import Variablefrom torch.utils.data import DataLoaderfrom torch.utils.tensorboard import SummaryWritersys.path.append("../pytorch_autoencoder")from utils.data_augmentation_fp import FingerprintDatasetfrom ae_model import autoencoderfrom ae_perf_loss import output_ae_imgs_if __name__ == '__main__':    num_epochs = 100000    batch_size = 128    learning_rate = 1e-3    img_width = 200    img_height = 200    model_width = 224    model_height = 224    pad_width = int((model_width - img_width) / 2)    pad_height = int((model_height - img_height) / 2)    np_pad = ((pad_height, model_height - img_height - pad_height), (pad_width, model_width - img_width - pad_width))    use_cuda = torch.cuda.is_available()    device = torch.device('cuda:0' if use_cuda else 'cpu')    if not os.path.exists('./dc_img'):        os.mkdir('./dc_img')    train_folder = "D:\\git\\Dev2930_NP"    test_folder = "D:\\git\\Dev2930_NP"    output_folder = train_folder + "_AE"    writer = SummaryWriter()    dataset = FingerprintDataset(root_dir=train_folder, csv_file='list.csv', img_width=img_width, img_height=img_height,                                 pad_width=np_pad, RBS=True)    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)    dataset_test = FingerprintDataset(root_dir=test_folder, csv_file='list_test.csv', img_width=img_width,                                      img_height=img_height,                                      pad_width=np_pad, RBS=True)    dataloader_test = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False)    model = autoencoder().to(device=device)    criterion = nn.MSELoss()    weighting_file = './conv_autoencoder_1900.pth'    if os.path.exists(weighting_file):        model.load_state_dict(torch.load(weighting_file))        model.eval()        model.to(device)    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,                                 weight_decay=1e-5)    # and a learning rate scheduler    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,                                                   step_size=1000,                                                   gamma=0.5)    N_TEST_IMG = 5    f, a = plt.subplots(3, N_TEST_IMG, figsize=(10, 6))    plt.ion()  # continuously plot    for epoch in range(num_epochs):        train_loss = 0        total_step = int(dataset.size / batch_size)        with progressbar.ProgressBar(max_value=total_step) as bar:            for step, (img, label) in enumerate(dataloader):                img = Variable(img).to(device=device)                label = Variable(label).to(device=device)                # ===================forward=====================                output = model(img)                output = (output - torch.min(output)) / (torch.max(output) - torch.min(output)) * 255                loss = criterion(output, label)                train_loss += loss.data                # ===================backward====================                optimizer.zero_grad()                loss.backward()                optimizer.step()                # print([x.grad for x in model.parameters()])                if step == 0:                    writer.add_scalar('Loss/train', loss.data, epoch)                    # show training result                    if epoch % 10 == 0:                        # plotting decoded image                        for i in range(N_TEST_IMG):                            a0 = np.reshape(img.data.cpu().numpy()[i], (model_width, model_height))                            a0 = a0[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                            a1 = np.reshape(label.data.cpu().numpy()[i], (model_width, model_height))                            a1 = a1[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                            a2 = np.reshape(output.data.cpu().numpy()[i], (model_width, model_height))                            a2 = a2[np_pad[0][0]:img_height, np_pad[1][0]:img_width]                            a[0][i].clear()                            a[0][i].imshow(a0, cmap='gray')                            a[1][i].clear()                            a[1][i].imshow(a1, cmap='gray')                            a[2][i].clear()                            a[2][i].imshow(a2, cmap='gray')                            pass                        plt.draw()                        # plt.pause(0.05)                        plt.savefig('./dc_img/image_{}.png'.format(epoch))                        pass                bar.update(step)            pass  # 1 train epoch end        writer.add_scalar('Loss/train', train_loss.data, epoch)        # test        output_ae_imgs_(test_folder, dataset_test, model, device)        # # test        # with torch.no_grad():        #     test_loss = 0        #     output_ = None        #     if epoch % 50 == 0:        #         for step_, (img_, label_) in enumerate(dataloader_test):        #             img_ = Variable(img_).to(device=device)        #             label_ = Variable(label_).to(device=device)        #        #             # ===================forward=====================        #             output_ = model(img_)        #             output_ = (output_ - torch.min(output_)) / (torch.max(output_) - torch.min(output_)) * 255        #             loss_ = criterion(output_, label_)        #             test_loss += loss_        #        #         # plotting decoded image        #         for i in range(N_TEST_IMG):        #             a0 = np.reshape(img.data.cpu().numpy()[i], (model_width, model_height))        #             a0 = a0[np_pad[0][0]:img_height, np_pad[1][0]:img_width]        #             a1 = np.reshape(label.data.cpu().numpy()[i], (model_width, model_height))        #             a1 = a1[np_pad[0][0]:img_height, np_pad[1][0]:img_width]        #             a2 = np.reshape(output.data.cpu().numpy()[i], (model_width, model_height))        #             a2 = a2[np_pad[0][0]:img_height, np_pad[1][0]:img_width]        #             a[0][i].clear()        #             a[0][i].imshow(a0, cmap='gray')        #             a[1][i].clear()        #             a[1][i].imshow(a1, cmap='gray')        #             a[2][i].clear()        #             a[2][i].imshow(a2, cmap='gray')        #             pass        #         plt.draw()        #         # plt.pause(0.05)        #         plt.savefig('./dc_img/image_{}_test.png'.format(epoch))        #        #         writer.add_scalar('Loss/test', test_loss.data, epoch)        #         writer.add_image('test', output_[0], epoch)        #         pass        lr_scheduler.step()        # ===================log========================        print('epoch [{}/{}], loss:{:.4f}, lr={}'              .format(epoch + 1, num_epochs, train_loss, optimizer.param_groups[0]['lr']))        if epoch % 100 == 0 and epoch > 0:            torch.save(model.state_dict(), './model/conv_autoencoder_{}.pth'.format(epoch))            print('save ./model/conv_autoencoder_{}.pth'.format(epoch))    writer.close()