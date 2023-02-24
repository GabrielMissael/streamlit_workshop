import numpy as np # linear algebra
import matplotlib.pyplot as plt # for plotting
from sklearn import datasets, svm, metrics # for machine learning
from sklearn.model_selection import train_test_split # for splitting data
import seaborn as sns # for plotting
from skimage.transform import resize # for resizing images

def show_resize(img):
    # Load custom image
    # img = plt.imread(path)

    img_resize = adapt_image(img)

    # Show resized and original image
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img, cmap=plt.cm.gray_r, interpolation='nearest')
    ax[0].set_title('Original')
    ax[1].imshow(img_resize, cmap=plt.cm.gray_r, interpolation='nearest')
    ax[1].set_title('Resized')

    # Remove ticks from the plot
    ax[0].set_xticks([])
    ax[0].set_yticks([])

    ax[1].set_xticks([])
    ax[1].set_yticks([])

    plt.show()

def adapt_image(img):
    img_resize = resize(img, (8, 8))

    # Convert to grayscale
    img_resize = np.mean(img_resize, axis=2)

    # Invert colors
    img_resize = 1 - img_resize

    # Scale to 0-16
    img_resize = img_resize * 16

    # Round to nearest integer
    img_resize = np.round(img_resize)

    return img_resize

def predict_digit(img, show = True):

    img_resize = adapt_image(img)

    # Flatten image
    img_resize = img_resize.reshape(1, -1)

    # Predict digit
    pred = clf.predict(img_resize)
    print('Predicción: %i' % pred[0])

    if show:
        show_resize(img)

# Function to show random example of a given digit
def show_digit(digit, digits):

    # Get indexes of all examples of given digit
    idx = np.where(digits.target == digit)[0]

    # Pick a random example
    i = np.random.choice(idx)

    # Return the image
    img = digits.images[i]

    return img
