from tensorflow.keras import preprocessing as preproc
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.models import Sequential as seq, load_model as load
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.callbacks import ModelCheckpoint, TerminateOnNaN
from tensorflow.keras.metrics import Accuracy, AUC, Precision, Recall, SpecificityAtSensitivity
from pandas import DataFrame as df

datagen = preproc.image.ImageDataGenerator(
    validation_split=.2,
    rescale=1./255,
    brightness_range=[25.5, 75.5],
    shear_range=0.3,
    zoom_range=0.2,
    horizontal_flip=True,
)

train = datagen.flow_from_directory(
    directory="data/plot_roi/",
    class_mode="categorical",
    color_mode="rgb",
    target_size=(299, 299),
    shuffle=True,
    interpolation="bilinear",
    seed=42,
    subset="training",
)

test = datagen.flow_from_directory(
    directory="data/plot_roi/",
    class_mode="categorical",
    color_mode="rgb",
    target_size=(299, 299),
    shuffle=True,
    interpolation="bilinear",
    seed=42,
    subset="validation",
)

mask_InceptionResNetV2_model = seq([
    Input(shape=(299, 299, 3)),
    InceptionResNetV2(include_top=False, weights="imagenet", input_shape=(299, 299, 3)),
    GlobalAveragePooling2D(),
    Dense(2, activation="softmax", kernel_regularizer="l2")
], name="MASK_InceptionResNetV2")

mask_InceptionResNetV2_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adadelta(learning_rate=1e-2), 
    metrics=["acc", Precision(.51), Recall(.51), SpecificityAtSensitivity(.5), AUC()]
)

callbacks = [
    ModelCheckpoint(filepath="ckpt/checkpoint-augment-l2-inceptionresnetv2-mask-{epoch:02d}-{val_acc:.3f}.h5", monitor="val_acc", save_best_only=True, mode="max"),
    TerminateOnNaN()
]

mask_InceptionResNetV2_model_result = mask_InceptionResNetV2_model.fit(
    x=train, validation_data=test, epochs=30, callbacks=callbacks)

mask_InceptionResNetV2_model.save("model/augment_l2_mask_InceptionResNetV2_model.h5")
mask_InceptionResNetV2_model.save_weights("model/augment_l2_mask_InceptionResNetV2_weights.h5")
df.from_dict(mask_InceptionResNetV2_model_result.history).to_csv('result/augment_l2_mask_InceptionResNetV2_model_result.csv', index=False)