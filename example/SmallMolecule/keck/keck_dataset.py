# Use DeepChem dataset loader structure to load keck dataset and featurize it.
"""
KECK dataset loader.
Added id_field name so that splited file can be joined back to external resource.
"""
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals

import os
import deepchem


def load_keck(featurizer='ECFP', split='random', reload=True):
  """Load KECK datasets. Does not do train/test split"""

  if "DEEPCHEM_DATA_DIR" in os.environ:
    data_dir = os.environ["DEEPCHEM_DATA_DIR"]
  else:
    data_dir = "/tmp"
  if reload:
    save_dir = os.path.join(data_dir, "keck/" + featurizer + "/" + split)

  dataset_file = os.path.join("/home/haozhen/Haozhen-data/Tonys_lab/working_session/dataset",
                                "keck_complete.csv")

  # Featurize KECK dataset
  print("About to featurize KECK dataset.")
  if featurizer == 'ECFP':
    featurizer = deepchem.feat.CircularFingerprint(size=1024)
  elif featurizer == 'GraphConv':
    featurizer = deepchem.feat.ConvMolFeaturizer()
  elif featurizer == 'Weave':
    featurizer = deepchem.feat.WeaveFeaturizer()
  elif featurizer == 'Raw':
    featurizer = deepchem.feat.RawFeaturizer()
  elif featurizer == 'Descriptors':
    featurizer = deepchem.feat.RDKitDescriptors()

  KECK_tasks = ['Keck_Pria_AS_Retest']

  if reload:
    loaded, all_dataset, transformers = deepchem.utils.save.load_dataset_from_disk(
        save_dir)
    if loaded:
      return KECK_tasks, all_dataset, transformers

  loader = deepchem.data.CSVLoader(
    tasks=KECK_tasks, smiles_field="SMILES", id_field="Molecule",featurizer=featurizer)

  dataset = loader.featurize(dataset_file)
  # Initialize transformers
  transformers = [
      deepchem.trans.BalancingTransformer(transform_w=True, dataset=dataset)
  ]

  print("About to transform data")
  for transformer in transformers:
    dataset = transformer.transform(dataset)

  splitters = {
      'index': deepchem.splits.IndexSplitter(),
      'random': deepchem.splits.RandomSplitter(),
      'scaffold': deepchem.splits.ScaffoldSplitter()
  }
  splitter = splitters[split]
  print("Performing new split.")
  train, valid, test = splitter.train_valid_test_split(dataset)

  if reload:
    deepchem.utils.save.save_dataset_to_disk(save_dir, train, valid, test,
                                             transformers)

  return KECK_tasks, (train, valid, test), transformers
