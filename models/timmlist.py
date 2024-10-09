from pprint import pprint

import timm

model_names = timm.list_models('*vit*')
pprint(model_names)
