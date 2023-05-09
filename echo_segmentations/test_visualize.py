import matplotlib.pyplot as plt
import torch


def visualize_test_inference(model_, data_loader, test_batch_size, num_classes, device):
    X, Y = next(iter(data_loader))
    Y = torch.nn.functional.one_hot(Y.long(), num_classes).permute(0, 3, 1, 2)
    X, Y = X.to(device), Y.to(device)
    Y_pred = model_(X)
    Y_pred = torch.argmax(Y_pred, dim=1)

    fig, axes = plt.subplots(test_batch_size, 3)

    cols = ['Target Image', 'Ground Truth Semantic Map', 'Predicted Semantic Map']
    for ax, col in zip(axes[0], cols):
        ax.set_title(col)

    for ax in axes.flat:
        ax.axis('off')

    for i in range(test_batch_size):
        target_img = (torch.squeeze(X[i]).detach().cpu().numpy() * 255).astype('uint8')
        gt_labelmap = torch.argmax(Y[i], dim=0).detach().cpu().numpy()
        predicted_labelmap = Y_pred[i].detach().cpu().numpy()

        axes[i, 0].imshow(target_img, vmin=0, vmax=255, cmap='gray')
        axes[i, 1].imshow(gt_labelmap)
        # predicted_labelmap[predicted_labelmap != 1] = 0
        axes[i, 2].imshow(predicted_labelmap)

    plt.show()
    return
