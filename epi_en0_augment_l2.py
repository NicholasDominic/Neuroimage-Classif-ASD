from tensorflow.keras import preprocessing as preproc
from tensorflow.keras.applications import EfficientNetB0
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
    directory="data/plot_anat/",
    class_mode="categorical",
    color_mode="rgb",
    target_size=(224, 224),
    shuffle=True,
    interpolation="bilinear",
    seed=42,
    subset="training",
)

test = datagen.flow_from_directory(
    directory="data/plot_anat/",
    class_mode="categorical",
    color_mode="rgb",
    target_size=(224, 224),
    shuffle=True,
    interpolation="bilinear",
    seed=42,
    subset="validation",
)

epi_EfficientNetB0_model = seq([
    Input(shape=(224, 224, 3)),
    EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3)),
    GlobalAveragePooling2D(),
    Dense(2, activation="softmax", kernel_regulizer="l2")
], name="EPI_EfficientNetB0")

epi_EfficientNetB0_model.compile(
    loss="categorical_crossentropy",
    optimizer=Adadelta(learning_rate=1e-2), 
    metrics=["acc", Precision(.51), Recall(.51), SpecificityAtSensitivity(.5), AUC()]
)

callbacks = [
    ModelCheckpoint(filepath="ckpt/checkpoint-augment-l2-efficientnetb0-epi-{epoch:02d}-{val_acc:.3f}.h5", monitor="val_acc", save_best_only=True, mode="max"),
    TerminateOnNaN()
]

epi_EfficientNetB0_model_result = epi_EfficientNetB0_model.fit(
    x=train, validation_data=test, epochs=30, callbacks=callbacks)

epi_EfficientNetB0_model.save("model/augment_l2_epi_EfficientNetB0_model.h5")
epi_EfficientNetB0_model.save_weights("model/augment_l2_epi_EfficientNetB0_weights.h5")
df.from_dict(epi_EfficientNetB0_model_result.history).to_csv('result/augment_l2_epi_EfficientNetB0_model_result.csv', index=False)