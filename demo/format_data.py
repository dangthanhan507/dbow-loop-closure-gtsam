import os

if __name__ == '__main__':
    filenames = os.listdir('images3/')
    print(filenames)
    
    filenames = sorted(filenames, key=lambda x: int(x.split('.')[0][-6:]))
    
    for filename in filenames:
        print(filename)
        
    #rename files to image + number
    for i, filename in enumerate(filenames):
        os.rename('images3/' + filename, 'images3/' + 'image' + str(i) + '.jpg')
    