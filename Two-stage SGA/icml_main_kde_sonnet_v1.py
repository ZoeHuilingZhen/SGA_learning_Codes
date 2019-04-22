from icml_model_sonnet_v1 import  *
#from icml_model_sonnet_v2 import  *
from scipy import stats, optimize, interpolate
#import tensorflow.contrib.eager as tfe
from matplotlib.pyplot import plot,savefig

def kde(mu, tau, i, mode, align, bbox=None, xlabel="", ylabel="", cmap='Blues'):
    values = np.vstack([mu, tau])
    print(values)
    kernel = sp.stats.gaussian_kde(values)

    fig, ax = plt.subplots()
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    xx, yy = np.mgrid[bbox[0]:bbox[1]:300j, bbox[2]:bbox[3]:300j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    f = np.reshape(kernel(positions).T, xx.shape)
    cfset = ax.contourf(xx, yy, f, cmap=cmap)
    plt.show()
    if align:
        print("Align--------------")
        savefig("./kde_%s_%d_v1_align.jpg" %(mode,i))
    else:
        print("Nonalign--------------")
        savefig("./kde_%s_%d_v1_nonalign.jpg" %(mode,i))


def train(train_op, x_fake, z, init, disc_loss, gen_loss, z_dim,mode, align, 
          n_iter=10001, n_save=2000):
    bbox = [-2, 2, -2, 2]
    batch_size = x_fake.get_shape()[0].value
    ztest = [np.random.randn(batch_size, z_dim) for i in range(10)]
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(init)

        for i in range(n_iter):
            disc_loss_out, gen_loss_out, _ = sess.run(
                [disc_loss, gen_loss, train_op])
            if i % n_save == 0:
                print('i = %d, discriminant loss = %.4f, generator loss =%.4f' %
                      (i, disc_loss_out, gen_loss_out))
                x_out = np.concatenate(
                    [sess.run(x_fake, feed_dict={z: zt}) for zt in ztest], axis=0)
                kde(x_out[:, 0], x_out[:, 1], i, mode, align, bbox=bbox)


def learn_mixture_of_gaussians(mode):
    print(mode)

    def x_real_builder(batch_size):
        sigma = 0.1
        skel = np.array([
            [1.50, 1.50],
            [1.50, 0.50],
            [1.50, -0.50],
            [1.50, -1.50],
            [0.50, 1.50],
            [0.50, 0.50],
            [0.50, -0.50],
            [0.50, -1.50],
            [-1.50, 1.50],
            [-1.50, 0.50],
            [-1.50, -0.50],
            [-1.50, -1.50],
            [-0.50, 1.50],
            [-0.50, 0.50],
            [-0.50, -0.50],
            [-0.50, -1.50],
        ])
        temp = np.tile(skel, (batch_size // 16 + 1, 1))
        mus = temp[0:batch_size, :]
        return mus + sigma * tf.random_normal([batch_size, 2]) * .2

    z_dim = 64
    train_op, x_fake, z, init, disc_loss, gen_loss = reset_and_build_graph(
        depth=6, width=384, x_real_builder=x_real_builder, z_dim=z_dim,
        batch_size=256, learning_rate=1e-4, mode=mode, align = False)

    train(train_op, x_fake, z, init, disc_loss, gen_loss, z_dim, mode, align=False)
#learn_mixture_of_gaussians("RMS")
#learn_mixture_of_gaussians("SGA")
#tfe.enable_eager_execution()
learn_mixture_of_gaussians("ICML")
