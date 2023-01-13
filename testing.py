import database as db
from matplotlib import pyplot as plt


db.load_images()
labels = db.imgs[:5, 1]
print(db.get_one_hot(labels))