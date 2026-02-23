# Experiment Changelog

## v2 (2026-02-19) — Training improvements: validation + patience

### Rationale
v1 achieved EER=3.51% on LA (vs paper's 2.87%) and EER=7.84% on In-the-Wild (vs 7.46%).
Analysis identified two root causes for the performance gap:
1. **Early stopping patience=1** — too aggressive; model stopped at epoch 3 (only 4 epochs)
2. **No validation set** — early stopping monitored training loss, not generalization loss
3. **Training loss as stopping criterion** — prone to overfitting signals

### Changes to `main.py`

#### Change 1: Enable validation data loader (lines 277-287)
**BEFORE:**
```python
# #define dev (validation) dataloader
#
# d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),is_train=False,is_eval=False)
#
# print('no. of validation trials',len(file_dev))
#
# dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,labels = d_label_dev,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_dev/'),algo=args.algo)
#
# dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
#
# del dev_set,d_label_dev
```

**AFTER:**
```python
# define dev (validation) dataloader
d_label_dev,file_dev = genSpoof_list( dir_meta =  os.path.join(args.protocols_path+'ASVspoof_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt'),is_train=False,is_eval=False)

print('no. of validation trials',len(file_dev))

dev_set = Dataset_ASVspoof2019_train(args,list_IDs = file_dev,labels = d_label_dev,base_dir = os.path.join(args.database_path+'ASVspoof2019_LA_dev/'),algo=args.algo)

dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)

del dev_set,d_label_dev
```

#### Change 2: Enable validation evaluation in training loop (line 302)
**BEFORE:**
```python
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device)
        #val_loss, val_acc = evaluate_accuracy(dev_loader, model, device)
```

**AFTER:**
```python
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device)
        val_loss, val_acc = evaluate_accuracy(dev_loader, model, device)
```

#### Change 3: Use validation loss for early stopping (line 304)
**BEFORE:**
```python
        if running_loss < best_val_loss:
```

**AFTER:**
```python
        if val_loss < best_val_loss:
```

#### Change 4: Increase early stopping patience from 1 to 10 (line 320)
**BEFORE:**
```python
        if patience_counter >= 1:
```

**AFTER:**
```python
        if patience_counter >= 10:
```

#### Change 5: Enable validation loss/accuracy TensorBoard logging (lines 312-313)
**BEFORE:**
```python
        #writer.add_scalar('val_loss',0, epoch)
        #writer.add_scalar('val_acc', 0, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {} -{}'.format(epoch,
                                                   running_loss,0,0))
```

**AFTER:**
```python
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('val_acc', val_acc, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - train_loss: {} - val_loss: {} - val_acc: {}'.format(epoch,
                                                   running_loss, val_loss, val_acc))
```

#### Change 6: Fix best_val_loss tracking variable (line 304)
**BEFORE (inherited bug from v1):**
```python
        if val_loss < best_val_loss:
            best_val_loss = running_loss  # BUG: tracked wrong loss
```

**AFTER:**
```python
        if val_loss < best_val_loss:
            best_val_loss = val_loss  # FIX: track validation loss correctly
```

#### Change 7: Add torch.no_grad() to evaluate_accuracy (OOM fix)
**BEFORE:**
```python
def evaluate_accuracy(dev_loader, model, device):
    ...
    model.eval()
    ...
    for batch_x, batch_y in tqdm(dev_loader):
        batch_out = model(batch_x)  # builds computation graph, wastes GPU memory
```

**AFTER:**
```python
def evaluate_accuracy(dev_loader, model, device):
    ...
    model.eval()
    ...
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dev_loader):
            batch_out = model(batch_x)  # no grad = much less GPU memory
```
Note: Original code was missing `torch.no_grad()` during validation, causing OOM after
first training epoch (12.88 GiB allocated + validation graph exceeded 15.57 GiB).

### Infrastructure changes
- Created `database/ASVspoof_LA_cm_protocols/` symlink to `/media/lab2208/ssd/df_tampering/ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols/`
  (provides ASVspoof2019.LA.cm.dev.trl.txt for validation loader)

### Expected impact
- More training epochs (patience=10 allows model to train longer before stopping)
- Better model selection (validation loss reflects generalization, not memorization)
- 24,844 validation trials from ASVspoof2019 LA dev set
- Estimated training time: ~29 min/epoch (19 min train + 10 min val), 5-10 hours total

### Actual results
- 27 epochs trained, early stop at epoch 26 (best val_loss at epoch 16)
- Best val_loss=0.000468, val_acc=99.99%
- **LA improved marginally: 3.51% → 3.47% EER**
- **DF degraded significantly: 2.14% → 3.75% EER**
- **In-the-Wild degraded: 7.84% → 12.67% EER**
- Conclusion: LA-based validation caused overfitting to LA domain, harming cross-domain generalization


---

## v1 (2026-02-17) — Baseline reproduction

### Configuration
- Trained on ASVspoof2019 LA train (25,380 utterances)
- No validation set (commented out in original code)
- Early stopping patience=1 on training loss
- RawBoost algo=3, lr=1e-6, batch_size=5, WCE loss

### Results
- 4 epochs total (stopped at epoch 3)
- Best model: epoch_2.pth (train loss = 0.000661)
- ASVspoof 2021 LA: EER = 3.51%, min t-DCF = 0.2701
- In-the-Wild: EER = 7.84%
- ASVspoof 2021 DF: EER = 2.14%

### Artifacts
- Archived in experiments/v1/
