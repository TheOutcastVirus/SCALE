

bundle_uri = 'https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo-0.12/isprs-potsdam-ss/model-bundle.zip'
from rastervision.pytorch_learner import SemanticSegmentationLearner

learner = SemanticSegmentationLearner.from_model_bundle(bundle_uri, training=False)

scene_id = 5631
image_uri = f'https://s3.amazonaws.com/azavea-research-public-data/raster-vision/examples/model-zoo-0.12/isprs-potsdam-ss/3_12_sample.tif'
#label_uri = f's3://spacenet-dataset/spacenet/SN2_buildings/train/AOI_2_Vegas/geojson_buildings/SN2_buildings_train_AOI_2_Vegas_geojson_buildings_img{scene_id}.geojson'

from rastervision.core.data import ClassConfig

class_config = ClassConfig(
    names=["Car", "Building", "Low Vegetation", "Tree", "Impervious", "Clutter", "null"],
    colors=["#ffff00", "#0000ff", "#00ffff", "#00ff00", "#ffffff", "#ff0000", "black"])
class_config.ensure_null_class()

from rastervision.core.data import StatsTransformer

stats_uri = 's3://azavea-research-public-data/raster-vision/examples/model-zoo-0.30/spacenet-vegas-buildings-ss/analyze/stats/train_scenes/stats.json'
stats_tf = StatsTransformer.from_stats_json(stats_uri)
stats_tf

from rastervision.pytorch_learner import SemanticSegmentationSlidingWindowGeoDataset

import albumentations as A

ds = SemanticSegmentationSlidingWindowGeoDataset.from_uris(
    class_config=class_config,
    image_uri=image_uri,
    image_raster_source_kw=dict(raster_transformers=[stats_tf]),
    size=325,
    stride=325,
    out_size=325,
)

from rastervision.core.data import SemanticSegmentationLabels

predictions = learner.predict_dataset(
    ds,
    raw_out=True,
    numpy_out=True,
    predict_kw=dict(out_shape=(325, 325)),
    progress_bar=True)

pred_labels = SemanticSegmentationLabels.from_predictions(
    ds.windows,
    predictions,
    smooth=True,
    extent=ds.scene.extent,
    num_classes=len(class_config))

scores = pred_labels.get_score_arr(pred_labels.extent)
from matplotlib import pyplot as plt

scores_building = scores[0]
scores_background = scores[1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
fig.tight_layout(w_pad=-2)
ax1.imshow(scores_building, cmap='plasma')
ax1.axis('off')
ax1.set_title('building')
ax2.imshow(scores_background, cmap='plasma')
ax2.axis('off')
ax2.set_title('background')
plt.show()

pred_labels.save(
    uri=f'./spacenet-vegas-buildings-ss/predict/{scene_id}',
    crs_transformer=ds.scene.raster_source.crs_transformer,
    class_config=class_config)