import cv2 as cv
import numpy as np
import copy
import random
from sklearn.cluster import KMeans


class KMeansTool:
    def __init__(self, img_path, k):
        self.image = cv.imread(img_path)
        self.shape = self.image.shape
        self.k = k

        self.image_data = self.image.reshape(-1, 3)
        print(f"image_data {self.image_data}")

    def k_means(self):
        dst_img = copy.deepcopy(self.image)
        H, W, C = self.shape
        N = self.image_data.shape[0]
        data = np.array(self.image_data, dtype=np.float64)
        index = np.zeros(shape=(N))
        center = self.center_init(data)
        while True:
            for i in range(N):
                dist = np.sqrt(np.sum(np.square(data[i] - center), axis=1))
                index[i] = np.argmin(dist, axis=0)

            new_center = np.zeros(shape=(self.k, C))
            for i in range(self.k):
                node_index = np.where(index == i)
                temp = np.mean(data[node_index], axis=0)
                print(temp)
                new_center[i] = temp
            if (center == new_center).all():
                break
            center = new_center

        print(f"k_means \n{center} \n{index}")

        label_dict = {}
        color = 255 / self.k
        for item in range(self.k):
            label_dict.update({item: [item * color] * 3})

        index = index.reshape((H, W))
        for x in range(H):
            for y in range(W):
                dst_img[x][y] = label_dict[index[x][y]]
        cv.imwrite(f"k_means.png", dst_img)

    def center_init(self, data):

        N, C = data.shape
        center = np.zeros(shape=(self.k, C))
        for i in range(self.k):
            while True:
                ind = random.randint(0, N - 1)
                if data[ind] not in center:
                    center[i] = data[ind]
                    break
        return center

    def k_means_sk_learn(self):
        dst_img = copy.deepcopy(self.image)
        H, W, C = self.shape
        clt = KMeans(n_clusters=self.k)
        clt.fit(self.image_data)

        print(f"k_means_sk_learn \n{clt.cluster_centers_} \n{clt.labels_}")

        index = np.array(clt.labels_)
        label = np.unique(index)
        label_dict = {}
        color = 255 / self.k
        for item in label:
            label_dict.update({item: [item * color] * 3})

        index = index.reshape((H, W))
        for x in range(H):
            for y in range(W):
                dst_img[x][y] = label_dict[index[x][y]]
        cv.imwrite(f"k_means_sk_learn.png", dst_img)


if __name__ == '__main__':
    k_means_tool = KMeansTool("test.jpeg", 10)
    k_means_tool.k_means()
    k_means_tool.k_means_sk_learn()

    # data = np.zeros(shape=[3, 4, 3])
    # n = 0
    # for i in range(3):
    #     for j in range(4):
    #         data[i, j] = 2 * n
    #         n += 1
    # cv.imwrite('test.png',data)
