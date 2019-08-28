# coding: utf-8

from gans_gray_scale_DNN import GANs

import matplotlib.pyplot as plt
import matplotlib.animation as animation


import os
import numpy as np

np.random.seed(1)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# load model & weight
gans = GANs()
gans.load("model_data/generator.json", "model_data/generator_weight.h5", "model_data/discriminato.json", "model_data/discriminator_weight.h5", "model_data/gan.json", "model_data/gan_weight.h5")

# animate
fig = plt.figure()
plt.axis('off')
noise = np.zeros((1, 100))
p = gans.generater_predict(noise)
r = p.reshape((1, 64, 64))[0]
im = plt.imshow(r, interpolation='nearest', cmap='gray')


def init():
    im.set_data(r)
    return [im]


a = gans.create_noise(1)
b = np.zeros((1, 100))


def animate(i):
    global a, b
    x = np.linspace(0, np.pi, 100)

    b = a * np.sin(2 * np.pi * (x - 0.02 * i)) * np.cos(1 * np.pi * (x - 0.01 * i)) * 0.7 + 0.2

    p = gans.generater_predict(b)
    r = p.reshape((1, 64, 64))[0]

    im.set_array(r)
    return [im]


anim = animation.FuncAnimation(fig, animate, init_func=init, frames=1000, interval=100, blit=True)

# plt.rcParams['animation.ffmpeg_path'] = '/usr/local/bin/ffmpeg'
anim.save('/sample/animation.mp4', fps=30, extra_args=['-vcodec', 'libx264'])

# HTML(anim.to_html5_video())
