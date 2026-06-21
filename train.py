import os
import torch
import yaml
import argparse
from core.dataset import MMDataLoader
from core.losses import MultimodalLoss_stage1
from core.losses import MultimodalLoss_stage2
from core.scheduler import get_scheduler
from core.utils import setup_seed, get_best_results
from models.recap import build_model
from core.metric import MetricsTop 
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
print(device)

parser = argparse.ArgumentParser() 
parser.add_argument('--config_file', type=str, default='') 
parser.add_argument('--seed', type=int, default=-1) 
parser.add_argument('--stage', type=str, choices=['completion', 'fusion_prediction'], default='fusion_prediction')
parser.add_argument('--time', type=str, default='') 
parser.add_argument('--stage1_ckpt', type=str, default='')
parser.add_argument('--missing_rate_eval_test', type=float, default=None) 
parser.add_argument('--batch_size', type=int, default=64) 
opt = parser.parse_args()
print(opt)
from datetime import datetime


def resolve_stage1_checkpoint(ckpt_root):
    if opt.stage1_ckpt:
        return opt.stage1_ckpt
    if opt.time:
        if opt.time.endswith('.pth'):
            return os.path.join(ckpt_root, opt.time)
        return os.path.join(ckpt_root, f'stage1_modules_{opt.time}.pth')
    raise ValueError("Please provide --stage1_ckpt or --time for stage 2 training.")


def main():
    best_valid_results, best_test_results = {}, {}
    loss_history = {}
    config_file = 'configs/train_mosi.yaml' if opt.config_file == '' else opt.config_file

    with open(config_file) as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    print(args)

    seed = args['base']['seed'] if opt.seed == -1 else opt.seed
    setup_seed(seed)
    print("seed is fixed to {}".format(seed))

    if opt.missing_rate_eval_test is not None:
        args['base']['missing_rate_eval_test'] = opt.missing_rate_eval_test
        print("train: ", args['base']['missing_rate_eval_test'])

    stage = opt.stage
    print(f"Training stage: {stage}")

    ckpt_root = os.path.join('ckpt', args['dataset']['datasetName'])
    if not os.path.exists(ckpt_root):
        os.makedirs(ckpt_root)
    print("ckpt root :", ckpt_root)

    model = build_model(args).to(device)

    dataLoader = MMDataLoader(args)


    loss_fn_stage1 = MultimodalLoss_stage1(args)
    loss_fn_stage2 = MultimodalLoss_stage2(args)

    metrics = MetricsTop(train_mode = args['base']['train_mode']).getMetics(args['dataset']['datasetName'])
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'stage1_modules_seed{seed}_{timestamp}.pth'
    
    if opt.stage == 'completion':
        print("===> Stage 1: Completion training")
        optimizer = torch.optim.AdamW(model.parameters(),
                                    lr=args['base']['lr'],
                                    weight_decay=args['base']['weight_decay'])
        scheduler_warmup = get_scheduler(optimizer, args)
        best_model_state = None
        best_loss = float('inf')
        for epoch in range(1, args['base']['n_epochs_stage1'] + 1):
            train_loss_dict = train(model, dataLoader['train'], optimizer, loss_fn_stage1, loss_fn_stage2, epoch, metrics, mode='completion')
            for key, value in train_loss_dict.items():
                if key not in loss_history:
                    loss_history[key] = []
                loss_history[key].append(value)            
            scheduler_warmup.step()

            current_loss = train_loss_dict["loss"]
            if current_loss < best_loss:
                best_loss = current_loss
                best_model_state = {'generator': model.generator.state_dict()}
        if best_model_state is not None:
            torch.save(best_model_state, os.path.join(ckpt_root, filename))
            print(f"Best model saved with loss_total: {best_loss:.4f}")
        print("===> Saved completion & discriminator after Stage 1")
     
    elif opt.stage == 'fusion_prediction':
        print("===> Stage 2: Fusion training")
        
        # Load the Stage 1 module
        stage1_ckpt = resolve_stage1_checkpoint(ckpt_root)
        checkpoint = torch.load(stage1_ckpt, map_location=device)
        print(f"Loaded stage 1 checkpoint: {stage1_ckpt}")
        model.generator.load_state_dict(checkpoint['generator'])

        for p in model.generator.parameters():
            p.requires_grad = False

        optimizer = torch.optim.AdamW(model.parameters(),
                            lr=args['base']['lr'],
                            weight_decay=args['base']['weight_decay'])
    
        # Recreate the learning rate scheduler
        scheduler_warmup = get_scheduler(optimizer, args)


        for epoch in range(1, args['base']['n_epochs_stage2']+1):
            train_loss_dict = train(model, dataLoader['train'], optimizer, loss_fn_stage1, loss_fn_stage2, epoch, metrics, mode=stage) 
            for key, value in train_loss_dict.items():
                if key not in loss_history:
                    loss_history[key] = []
                loss_history[key].append(value)
            # train(model, dataLoader['train'], optimizer, loss_fn, epoch, metrics)

            if args['base']['do_validation']:
                valid_results = evaluate(model, dataLoader['valid'], loss_fn_stage2, epoch, metrics)
                best_valid_results = get_best_results(valid_results, best_valid_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=False)
                print(f'Current Best Valid Results: {best_valid_results}')

            test_results = evaluate(model, dataLoader['test'], loss_fn_stage2, epoch, metrics)
            best_test_results = get_best_results(test_results, best_test_results, epoch, model, optimizer, ckpt_root, seed, save_best_model=True)
            print(f'Current Best Test Results: {best_test_results}\n')

            scheduler_warmup.step()


def train(model, train_loader, optimizer, loss_fn_stage1, loss_fn_stage2, epoch, metrics, mode='completion'):
    y_pred, y_true = [], []
    loss_dict = {}
    results = {}

    model.train()
    for cur_iter, data in enumerate(train_loader):
        complete_input = (data['vision'].to(device), data['audio'].to(device), data['text'].to(device))
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        label = {'sentiment_labels': sentiment_labels}

        if mode == 'completion':
            out = model(complete_input, incomplete_input, sentiment_labels, mode='completion')
            loss_stage1 = loss_fn_stage1(out, label)
            loss_stage1['loss'].backward()
            optimizer.step()
            optimizer.zero_grad()

            if cur_iter == 0:
                for key, value in loss_stage1.items():
                    loss_dict[key] = value.item() if isinstance(value, torch.Tensor) else value
            else:
                for key, value in loss_stage1.items():
                    # loss_dict[key] += value.item()
                    loss_dict[key] += value.item() if isinstance(value, torch.Tensor) else value
            loss_dict = {key: value / (cur_iter+1) for key, value in loss_dict.items()}

        else:
            out = model(complete_input, incomplete_input, sentiment_labels, mode='fusion_prediction')
            loss_stage2 = loss_fn_stage2(out, label)
            optimizer.zero_grad()
            loss_stage2['loss'].backward()
            optimizer.step()
            
            y_pred.append(out['sentiment_preds'].cpu())
            y_true.append(label['sentiment_labels'].cpu())

            if cur_iter == 0:
                for key, value in loss_stage2.items():
                    loss_dict[key] = value.item() if isinstance(value, torch.Tensor) else value
            else:
                for key, value in loss_stage2.items():
                    # loss_dict[key] += value.item()
                    loss_dict[key] += value.item() if isinstance(value, torch.Tensor) else value

            pred, true = torch.cat(y_pred), torch.cat(y_true)
            results = metrics(pred, true)

            loss_dict = {key: value / (cur_iter+1) for key, value in loss_dict.items()}

    print(f'Train Loss Epoch {epoch}: {loss_dict}')
    print(f'Train Results Epoch {epoch}: {results}')

    return loss_dict

def evaluate(model, eval_loader, loss_fn_stage2, epoch, metrics):
    loss_dict = {}

    y_pred, y_true = [], []

    model.eval()
    
    for cur_iter, data in enumerate(eval_loader):
        complete_input = (None, None, None)
        incomplete_input = (data['vision_m'].to(device), data['audio_m'].to(device), data['text_m'].to(device))

        sentiment_labels = data['labels']['M'].to(device)
        label = {'sentiment_labels': sentiment_labels}
        
        with torch.no_grad():
            out = model(complete_input, incomplete_input, sentiment_labels, mode='fusion_prediction')

        loss = loss_fn_stage2(out, label)

        y_pred.append(out['sentiment_preds'].cpu())
        y_true.append(label['sentiment_labels'].cpu())

        if cur_iter == 0:
            for key, value in loss.items():
                try:
                    loss_dict[key] = value.item()
                except:
                    loss_dict[key] = value
        else:
            for key, value in loss.items():
                try:
                    loss_dict[key] += value.item() 
                except:
                    loss_dict[key] += value
    
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    results = metrics(pred, true)

    return results


if __name__ == '__main__':
    main()
