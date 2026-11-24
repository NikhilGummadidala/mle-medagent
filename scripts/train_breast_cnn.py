"""Local CNN section of BreastCancerAgent.ipynb, with auditable CBIS-DDSM inputs.

The architecture is imported from the existing inference module. No MIAS images
are used. New training weights are not represented as the team's old checkpoint.
"""
from __future__ import annotations
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from models.breast_cancer import BreastCancerCNN, _preprocess_image

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def write_json(path, data):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False))
    temp.replace(path)


def prepare(dataset_root: str, run_dir: str):
    source, out = Path(dataset_root).resolve(), Path(run_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cache = out / 'images_224'; cache.mkdir(exist_ok=True)
    # dicom_info describes image files but has no pathology label. Join it to the
    # four pathology case tables; using the first arbitrary CSV is incorrect.
    dicom = pd.read_csv(source / 'csv/dicom_info.csv')
    full = dicom[dicom.SeriesDescription == 'full mammogram images'].copy()
    assert full.SeriesInstanceUID.is_unique
    indexed = full.set_index('SeriesInstanceUID')
    cases = pd.concat([pd.read_csv(p).assign(source_csv=p.name)
                       for p in sorted((source/'csv').glob('*case_description*.csv'))], ignore_index=True)
    grouped = defaultdict(list)
    missing = []
    for row in cases.to_dict('records'):
        parts = str(row['image file path']).strip().replace('\\', '/').split('/')
        uid = parts[-2]
        label = {'BENIGN': 0, 'BENIGN_WITHOUT_CALLBACK': 0, 'MALIGNANT': 1}.get(row['pathology'])
        if label is None:
            raise ValueError('Unknown pathology: ' + str(row['pathology']))
        if uid not in indexed.index:
            missing.append(dict(patient_id=row['patient_id'], original_path=row['image file path'],
                                source_csv=row['source_csv'], reason='No matching full-mammogram series in published JPG metadata'))
            continue
        record = indexed.loc[uid]
        assert str(record.PatientID) == parts[0], 'Series/exam ID mismatch'
        image_path = source / str(record.image_path).split('CBIS-DDSM/', 1)[1]
        assert image_path.is_file(), f'Missing published image: {image_path}'
        grouped[str(image_path)].append(dict(patient_id=row['patient_id'], label=label,
                                             pathology=row['pathology'], source_csv=row['source_csv'], series_uid=uid))
    rows = []
    # Decode independent images concurrently; map preserves sorted deterministic order.
    def decode(item):
        path, labels = item
        patients = {r['patient_id'] for r in labels}
        assert len(patients) == 1, f'Image has conflicting patient IDs: {path}'
        label = max(r['label'] for r in labels)
        with Image.open(path) as im:
            rgb = im.convert('RGB')
            original_size = rgb.size
            pixels_hash = hashlib.sha256(np.asarray(rgb).tobytes()).hexdigest()
            resized = rgb.resize((224, 224), Image.Resampling.BICUBIC)
            assert np.asarray(resized).std() > 0, 'Constant image is not a valid training example'
            target = cache / (hashlib.sha256(path.encode()).hexdigest()[:24] + '.png')
            if target.exists():
                with Image.open(target) as existing:
                    assert np.array_equal(np.asarray(existing), np.asarray(resized)), 'Cached image mismatch'
            else:
                resized.save(target)
        return dict(patient_id=next(iter(patients)), image_path=str(target), original_image=path,
                    label=label, pathology_rows=json.dumps(labels), pixel_sha256=pixels_hash,
                    original_width=original_size[0], original_height=original_size[1])
    seen_pixels = {}
    duplicates = []
    conflicts = []
    conflicted_hashes = set()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for i, row in enumerate(pool.map(decode, sorted(grouped.items()))):
            pixel_hash = row['pixel_sha256']
            if pixel_hash in seen_pixels:
                existing = rows[seen_pixels[pixel_hash]]
                pair = dict(path=row['original_image'], retained_path=existing['original_image'],
                            patient_id=row['patient_id'], retained_patient_id=existing['patient_id'],
                            label=row['label'], retained_label=existing['label'], pixel_sha256=pixel_hash)
                if existing['label'] != row['label'] or pixel_hash in conflicted_hashes:
                    conflicted_hashes.add(pixel_hash)
                    conflicts.append(pair)
                else:
                    duplicates.append(pair)
            else:
                seen_pixels[pixel_hash] = len(rows)
                rows.append(row)
            if (i+1) % 200 == 0:
                write_json(out/'preparation_progress.json',dict(processed=i+1,total=len(grouped),conflicting_pixel_groups=len(conflicted_hashes)))
                print(f'Prepared {i+1}/{len(grouped)} full mammograms', flush=True)
    # Remove every copy of ambiguous exact-pixel groups, rather than guessing a label.
    rows = [row for row in rows if row['pixel_sha256'] not in conflicted_hashes]
    write_json(out/'excluded_conflicting_pixel_groups.json', conflicts)
    data = pd.DataFrame(rows)
    patients = data.groupby('patient_id').label.max().sort_index()
    trainval, test = train_test_split(patients.index, test_size=.15, random_state=SEED, stratify=patients.values)
    train, val = train_test_split(trainval, test_size=.15/.85, random_state=SEED, stratify=patients.loc[trainval].values)
    split_for = {p: name for name, ids in [('train',train),('validation',val),('test',test)] for p in ids}
    data['split'] = data.patient_id.map(split_for)
    assert set(train).isdisjoint(val) and set(train).isdisjoint(test) and set(val).isdisjoint(test)
    assert data.groupby('pixel_sha256').split.nunique().max() == 1
    data.to_csv(out/'split_manifest.csv', index=False)
    write_json(out/'excluded_metadata_rows.json', missing)
    write_json(out/'exact_duplicates.json', duplicates)
    summary = dict(dataset='CBIS-DDSM', kaggle_ref='awsaf49/cbis-ddsm-breast-cancer-image-dataset/versions/1',
                   dataset_root=str(source), metadata_rows=len(cases), matched_unique_full_images=len(grouped),
                   excluded_unmatched_rows=len(missing), excluded_exact_duplicates=len(duplicates), excluded_conflicting_pixel_groups=len(conflicted_hashes),
                   used_images=len(data), patients=int(data.patient_id.nunique()),
                   split_images=data.groupby('split').size().to_dict(),
                   split_patients=data.groupby('split').patient_id.nunique().to_dict(),
                   split_class_counts=data.groupby(['split','label']).size().unstack(fill_value=0).to_dict(),
                   mias_images_used=0, seed=SEED,
                   split_policy='Patient-level stratified approximately 70/15/15 across published full images; original source split preserved in row provenance but NOT an official CBIS test benchmark',
                   preprocessing='RGB, PIL BICUBIC resize 224x224 to match existing inference; ImageNet mean/std; full field of view, no ROI labels as inputs',
                   label_policy='BENIGN and BENIGN_WITHOUT_CALLBACK=0, MALIGNANT=1; any malignant abnormality within an image makes the image positive',
                   original_notebook_sha256=hashlib.sha256((ROOT/'diag-agents/BreastCancerAgent.ipynb').read_bytes()).hexdigest(),
                   source_citations=['https://doi.org/10.7937/K9/TCIA.2016.7O02S9CY', 'https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset'])
    write_json(out/'data_summary.json', summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


class Mammograms(Dataset):
    def __init__(self, table, training=False):
        self.paths = table.image_path.tolist()
        self.labels = table.label.astype(float).tolist()
        # Cache compact RGB uint8 pixels so large original JPGs are never decoded per epoch.
        self.images = [np.array(Image.open(p).convert('RGB')) for p in self.paths]
        augment = [transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
                   transforms.RandomRotation(15), transforms.ColorJitter(brightness=.2,contrast=.2)] if training else []
        self.transform = transforms.Compose(augment + [transforms.ToTensor(),
                  transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    def __len__(self): return len(self.labels)
    def __getitem__(self, index):
        return self.transform(Image.fromarray(self.images[index])), torch.tensor([self.labels[index]],dtype=torch.float32)


def make_loaders(out, batch_size=32):
    table = pd.read_csv(Path(out)/'split_manifest.csv')
    loaders = {}
    for split in ['train','validation','test']:
        dataset = Mammograms(table[table.split==split], training=split=='train')
        loaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=split=='train', num_workers=0)
    # Verify evaluation transform numerically matches the deployed predictor's preprocessing.
    expected = _preprocess_image(loaders['validation'].dataset.paths[0])[0]
    actual = loaders['validation'].dataset[0][0]
    assert torch.allclose(expected, actual, atol=1e-6), 'Training/inference preprocessing mismatch'
    return loaders


def evaluate(model, loader, device):
    model.eval(); probabilities=[]; labels=[]; total_loss=0
    with torch.no_grad():
        for images, target in loader:
            images, target = images.to(device), target.to(device)
            output = model(images)
            total_loss += nn.functional.binary_cross_entropy(output, target).item()*len(images)
            probabilities.extend(output.cpu().numpy().ravel().tolist())
            labels.extend(target.cpu().numpy().ravel().tolist())
    p = np.asarray(probabilities); y=np.asarray(labels)
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    return dict(loss=total_loss/len(y), accuracy=float(accuracy_score(y,p>=.5)),
                auroc=float(roc_auc_score(y,p)), confusion_matrix=confusion_matrix(y,p>=.5).tolist(),
                n=len(y)), probabilities, labels


def train(run_dir, max_epochs=50):
    out=Path(run_dir);random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    torch.set_num_threads(4)
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    loaders=make_loaders(out)
    model=BreastCancerCNN().to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-4)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,patience=5,factor=.5)
    best_loss=float('inf');best_epoch=None;stale=0;history=[]
    started=time.monotonic()
    print(f'Training original BreastCancerCNN on {device}; {max_epochs} maximum epochs',flush=True)
    for epoch in range(1,max_epochs+1):
        model.train();epoch_started=time.monotonic();loss_sum=0
        for batch,(images,labels) in enumerate(loaders['train']):
            images,labels=images.to(device),labels.to(device)
            optimizer.zero_grad();predictions=model(images)
            loss=nn.functional.binary_cross_entropy(predictions,labels)
            if not torch.isfinite(loss):raise RuntimeError('Non-finite training loss')
            loss.backward();optimizer.step();loss_sum+=loss.item()*len(images)
            if (batch+1)%20==0:
                write_json(out/'progress.json',dict(epoch=epoch,batch=batch+1,total_batches=len(loaders['train']),elapsed_seconds=time.monotonic()-started))
        val,_,_=evaluate(model,loaders['validation'],device)
        scheduler.step(val['loss'])
        if val['loss']<best_loss:
            best_loss=val['loss'];best_epoch=epoch;stale=0
            state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            torch.save(state,out/'best_weights.pt')
        else:stale+=1
        entry=dict(epoch=epoch,train_loss=loss_sum/len(loaders['train'].dataset),validation=val,
                   learning_rate=optimizer.param_groups[0]['lr'],seconds=time.monotonic()-epoch_started,best_epoch=best_epoch)
        history.append(entry);write_json(out/'history.json',history)
        write_json(out/'progress.json',dict(epoch=epoch,completed=True,best_epoch=best_epoch,elapsed_seconds=time.monotonic()-started))
        print(json.dumps(entry),flush=True)
        if stale>=10:
            print('Early stopping: no validation-loss improvement for 10 epochs',flush=True);break
    model.load_state_dict(torch.load(out/'best_weights.pt',map_location=device,weights_only=True),strict=True)
    test,probabilities,labels=evaluate(model,loaders['test'],device)
    test_table=pd.read_csv(out/'split_manifest.csv');test_table=test_table[test_table.split=='test'].copy()
    test_table['malignant_probability']=probabilities;test_table['true_label']=labels
    test_table.to_csv(out/'heldout_test_predictions.csv',index=False)
    write_json(out/'training_result.json',dict(architecture='models.breast_cancer.BreastCancerCNN',device=str(device),
              torch_version=torch.__version__,max_epochs=max_epochs,epochs_completed=len(history),best_epoch=best_epoch,
              selected_by='minimum validation binary cross entropy only',best_validation_loss=best_loss,
              hyperparameters=dict(seed=SEED,batch_size=32,learning_rate=1e-4,weight_decay=1e-4,optimizer='Adam',early_stopping_patience=10),
              heldout_test=test,elapsed_seconds=time.monotonic()-started,
              confidence_definition='max(p_malignant, 1-p_malignant); uncalibrated class score, not diagnostic accuracy or calibrated patient risk',
              limitations='New locally trained research checkpoint, not team original weights. No clinical validation. MIAS is a separate out-of-domain evaluation dataset. MPS numerical determinism not guaranteed.'))
    print('HELDOUT TEST '+json.dumps(test),flush=True)
    return test


def export(run_dir):
    out=Path(run_dir);result=json.loads((out/'training_result.json').read_text())
    weights=out/'best_weights.pt'
    state=torch.load(weights,map_location='cpu',weights_only=True)
    verified=BreastCancerCNN();verified.load_state_dict(state,strict=True);verified.eval()
    table=pd.read_csv(out/'split_manifest.csv');sample=table[table.split=='test'].iloc[0]
    with torch.no_grad():p=float(verified(_preprocess_image(sample.image_path)).item())
    assert 0<=p<=1 and np.isfinite(p)
    destination=ROOT/'saved_models/breast_cancer_cnn.pt'
    if destination.exists():raise FileExistsError(f'Will not overwrite an existing checkpoint: {destination}')
    temporary=destination.with_suffix('.pt.tmp');torch.save(state,temporary);temporary.replace(destination)
    digest=hashlib.sha256(destination.read_bytes()).hexdigest()
    provenance=dict(result,weights_sha256=digest,weights_path=str(destination),
                    created_at=datetime.now().astimezone().isoformat(),run_directory=str(out),
                    data=json.loads((out/'data_summary.json').read_text()),verification_sample=str(sample.image_path),
                    verification_malignant_probability=p,strict_state_dict_load=True)
    write_json(destination.with_suffix('.metadata.json'),provenance)
    write_json(out/'export.json',provenance)
    print(f'EXPORTED {destination} SHA256={digest}',flush=True)
    return str(destination)
