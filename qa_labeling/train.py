import random
import os
import torch
from pathlib import Path
import pandas as pd
import gc
import numpy as np
from transformers import BertTokenizer, BertConfig, get_cosine_schedule_with_warmup
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, RandomSampler
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch.nn as nn
import time
from tqdm import tqdm

from dataset import QuestDataset
from config import PipeLineConfig
from utils import compute_input_arays, compute_output_arrays, bcolors
from model import CustomBert
from infer import predict_result

ROOT = Path("../data_raw/")
target_cols = ['question_asker_intent_understanding', 'question_body_critical', 
               'question_conversational', 'question_expect_short_answer', 
               'question_fact_seeking', 'question_has_commonly_accepted_answer', 
               'question_interestingness_others', 'question_interestingness_self', 
               'question_multi_intent', 'question_not_really_a_question', 
               'question_opinion_seeking', 'question_type_choice',
               'question_type_compare', 'question_type_consequence',
               'question_type_definition', 'question_type_entity', 
               'question_type_instructions', 'question_type_procedure', 
               'question_type_reason_explanation', 'question_type_spelling', 
               'question_well_written', 'answer_helpful',
               'answer_level_of_information', 'answer_plausible', 
               'answer_relevance', 'answer_satisfaction', 
               'answer_type_instructions', 'answer_type_procedure', 
               'answer_type_reason_explanation', 'answer_well_written']

def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

config_1 = PipeLineConfig(3e-5,0.05,4,1,42,'uncased_1',True,False,0.7,0.3,8,False) ## These are experiments . You can do as much as you want as long as inference is faster 
config_2 = PipeLineConfig(4e-5,0.03,4,6,2019,'uncased_2',True,False,0.8,0.2,5,False)## Adding various different seeds , folds and learning rate and mixing them and then doing inference .
config_3 = PipeLineConfig(4e-5,0.03,4,4,2019,'small_test_3',True ,False, 0.8,0.2,3,True) ## For Small tests in Kaggle , less number of fold , less number of epochs.
config_4 = PipeLineConfig(4e-5,0.05,1,4,2019,'small_test_4',True ,False, 0.8,0.2,3,True)
## I am doing first experiement
config = config_3

def train_model(model, device, train_loader, optimizer, criterion, scheduler, config):
    
    model.train()
    avg_loss = 0.
    avg_loss_1 = 0.
    avg_loss_2 =0.
    avg_loss_3 =0.
    avg_loss_4 =0.
    avg_loss_5 =0.
   # tk0 = tqdm(enumerate(train_loader),total =len(train_loader))
    optimizer.zero_grad()

    for idx, batch in tqdm(enumerate(train_loader)):
        
        input_ids, input_masks, input_segments, labels, _ = batch
        input_ids, input_masks, input_segments, labels = input_ids.to(device), input_masks.to(device), input_segments.to(device), labels.to(device)            
        
        output_train = model(input_ids = input_ids.long(),
                             labels = None,
                             attention_mask = input_masks,
                             token_type_ids = input_segments,
                            )
        logits = output_train[0] #output preds
        loss1 = criterion(logits[:,0:9], labels[:,0:9])
        loss2 = criterion(logits[:,9:10], labels[:,9:10])
        loss3 = criterion(logits[:,10:21], labels[:,10:21])
        loss4 = criterion(logits[:,21:26], labels[:,21:26])
        loss5 = criterion(logits[:,26:30], labels[:,26:30])
        loss = config.question_weight*loss1+config.answer_weight*loss2+config.question_weight*loss3+config.answer_weight*loss4+config.question_weight*loss5
        #loss =(config.question_weight*criterion(logits[:,0:21], labels[:,0:21]) + config.answer_weight*criterion(logits[:,21:30], labels[:,21:30]))/config.accum_steps
        loss.backward()
        if (idx + 1) % config.accum_steps == 0:    
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        
        avg_loss += loss.item() / (len(train_loader)*config.accum_steps)
        avg_loss_1 += loss1.item() / (len(train_loader)*config.accum_steps)
        avg_loss_2 += loss2.item() / (len(train_loader)*config.accum_steps)
        avg_loss_3 += loss3.item() / (len(train_loader)*config.accum_steps)
        avg_loss_4 += loss4.item() / (len(train_loader)*config.accum_steps)
        avg_loss_5 += loss5.item() / (len(train_loader)*config.accum_steps)
        del input_ids, input_masks, input_segments, labels

    torch.cuda.empty_cache()
    gc.collect()
    return avg_loss ,avg_loss_1,avg_loss_2,avg_loss_3,avg_loss_4,avg_loss_5

def val_model(model, device, criterion, val_loader, val_shape, batch_size=8):

    avg_val_loss = 0.
    model.eval() # eval mode
    
    valid_preds = np.zeros((val_shape, len(target_cols)))
    original = np.zeros((val_shape, len(target_cols)))
    
    #tk0 = tqdm(enumerate(val_loader))
    with torch.no_grad():
        
        for idx, batch in enumerate(val_loader):
            input_ids, input_masks, input_segments, labels, _ = batch
            input_ids, input_masks, input_segments, labels = input_ids.to(device), input_masks.to(device), input_segments.to(device), labels.to(device)            
            
            output_val = model(input_ids = input_ids.long(),
                             labels = None,
                             attention_mask = input_masks,
                             token_type_ids = input_segments,
                            )
            logits = output_val[0] #output preds
            
            avg_val_loss += criterion(logits, labels).item() / len(val_loader)
            valid_preds[idx*batch_size : (idx+1)*batch_size] = logits.detach().cpu().squeeze().numpy()
            original[idx*batch_size : (idx+1)*batch_size]    = labels.detach().cpu().squeeze().numpy()
        
        score = 0
        preds = torch.sigmoid(torch.tensor(valid_preds)).numpy()
        
        # np.save("preds.npy", preds)
        # np.save("actuals.npy", original)
        
        rho_val = np.mean([spearmanr(original[:, i], preds[:,i]).correlation for i in range(preds.shape[1])])
        print('\r val_spearman-rho: %s' % (str(round(rho_val, 5))), end = 100*' '+'\n')
        
        for i in range(len(target_cols)):
            print(i, spearmanr(original[:,i], preds[:,i]))
            score += np.nan_to_num(spearmanr(original[:, i], preds[:, i]).correlation)
    return avg_val_loss, score/len(target_cols)

def main():
    seed_everything(config.seed)
    train = pd.read_csv(ROOT / 'train.csv')
    test = pd.read_csv(ROOT / 'test.csv')
    
    ## Get the shape of the data
    train_len, test_len = len(train.index), len(test.index)
    print(f'train size: {train_len}, test size: {test_len}')

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", do_lower_case=True)
    input_categories = list(train.columns[[1,2,5]])

    bert_config = BertConfig.from_pretrained('bert-base-uncased')
    bert_config.num_labels = len(target_cols)


    bert_model = 'bert-base-uncased'
    do_lower_case = 'uncased' in bert_model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    output_model_file = 'bert_pytorch.pt'


    test_inputs = compute_input_arays(test, input_categories, tokenizer, config, max_sequence_length=512,t_max_len=30, q_max_len=239, a_max_len=239)
    lengths_test = np.argmax(test_inputs[0] == 0, axis=1)
    lengths_test[lengths_test == 0] = test_inputs[0].shape[1]

    print(do_lower_case, bert_model, device, output_model_file)
    print(test_inputs)

    test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
    test_loader  = DataLoader(test_set, batch_size=32, shuffle=False)
    result = np.zeros((len(test), len(target_cols)))

    # test_inputs = compute_input_arays(test, input_categories, tokenizer, max_sequence_length=290,t_max_len=30, q_max_len=128, a_max_len=128)
    # lengths_test = np.argmax(test_inputs[0] == 0, axis=1)
    # lengths_test[lengths_test == 0] = test_inputs[0].shape[1]

    # print(do_lower_case, bert_model, device, output_model_file)
    # print(test_inputs)

    # test_set1 = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
    # test_loader1  = DataLoader(test_set, batch_size=32, shuffle=False)
    # result1 = np.zeros((len(test), len(target_cols)))

    NUM_FOLDS = config.fold  # change this
    SEED = config.seed
    BATCH_SIZE = 8
    epochs = config.epochs   # change this
    ACCUM_STEPS = 1

    kf = MultilabelStratifiedKFold(n_splits = NUM_FOLDS) # random_state = SEED

    #test_set = QuestDataset(inputs=test_inputs, lengths=lengths_test, labels=None)
    #test_loader  = DataLoader(test_set, batch_size=32, shuffle=False)
    #result = np.zeros((len(test), len(target_cols)))

    y_train = train[target_cols].values # dummy

    print(bcolors.FAIL, f"For Every Fold, Train {epochs} Epochs", bcolors.ENDC)
    if config.train :
        for fold, (train_index, val_index) in enumerate(kf.split(train.values, y_train)):
            # if fold > 0 : ## Saving GPU
            #     break 
            print(bcolors.HEADER, "Current Fold:", fold, bcolors.ENDC)

            train_df, val_df = train.iloc[train_index], train.iloc[val_index]
            print("Train and Valid Shapes are", train_df.shape, val_df.shape)
        
            print(bcolors.HEADER, "Preparing train datasets....", bcolors.ENDC)
        
            inputs_train = compute_input_arays(train_df, input_categories, tokenizer, config, max_sequence_length=290)
            outputs_train = compute_output_arrays(train_df, columns = target_cols)
            outputs_train = torch.tensor(outputs_train, dtype=torch.float32)
            lengths_train = np.argmax(inputs_train[0] == 0, axis=1)
            lengths_train[lengths_train == 0] = inputs_train[0].shape[1]
        
            print(bcolors.HEADER, "Preparing Valid datasets....", bcolors.ENDC)
        
            inputs_valid = compute_input_arays(val_df, input_categories, tokenizer, config, max_sequence_length=290)
            outputs_valid = compute_output_arrays(val_df, columns = target_cols)
            outputs_valid = torch.tensor(outputs_valid, dtype=torch.float32)
            lengths_valid = np.argmax(inputs_valid[0] == 0, axis=1)
            lengths_valid[lengths_valid == 0] = inputs_valid[0].shape[1]
        
            print(bcolors.HEADER, "Preparing Dataloaders Datasets....", bcolors.ENDC)

            train_set    = QuestDataset(inputs=inputs_train, lengths=lengths_train, labels=outputs_train)
            train_sampler = RandomSampler(train_set)
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,sampler=train_sampler)
        
            valid_set    = QuestDataset(inputs=inputs_valid, lengths=lengths_valid, labels=outputs_valid)
            valid_loader = DataLoader(valid_set, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
        
            model = CustomBert(config=bert_config)
            model.zero_grad()
            model.to(device)
            torch.cuda.empty_cache()
            if config.freeze : ## This is basically using out of the box bert model while training only the classifier head with our data . 
                for param in model.bert.parameters():
                    param.requires_grad = False
            model.train()
        
            i = 0
            best_avg_loss   = 100.0
            best_score      = -1.
            best_param_loss = None
            best_param_score = None
            param_optimizer = list(model.named_parameters())
            no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
            optimizer_grouped_parameters = [
                {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.8},
                {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
                ]        

            optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=config.lr, eps=4e-5)
        # optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, eps=4e-5)
            criterion = nn.BCEWithLogitsLoss()
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=config.warmup, num_training_steps= epochs*len(train_loader)//ACCUM_STEPS)
            print("Training....")
        
            for epoch in range(config.epochs):

                torch.cuda.empty_cache()
            
                start_time   = time.time()
                avg_loss,avg_loss_1,avg_loss_2 ,avg_loss_3, avg_loss_4, avg_loss_5   = train_model(model, device, train_loader, optimizer, criterion, scheduler, config)
                avg_val_loss, score = val_model(valid_loader, val_shape=val_df.shape[0])
                elapsed_time = time.time() - start_time

                print(bcolors.OKGREEN, 'Epoch {}/{} \t loss={:.4f} \t val_loss={:.4f} \t train_loss={:.4f} \t train_loss_1={:.4f} \t train_loss_2={:.4f} \t train_loss_3={:.4f} \t train_loss_4={:.4f}  \t train_loss_5={:.4f} \t score={:.6f} \t time={:.2f}s'.format(
                    epoch + 1, epochs, avg_loss, avg_val_loss,avg_loss,avg_loss_1,avg_loss_2,avg_loss_3,avg_loss_4,avg_loss_5, score, elapsed_time),
                bcolors.ENDC
                )
                if best_avg_loss > avg_val_loss:
                    i = 0
                    best_avg_loss = avg_val_loss 
                    best_param_loss = model.state_dict()

                if best_score < score:
                    best_score = score
                    best_param_score = model.state_dict()
                    print('best_param_score_{}_{}.pt'.format(config.expname ,fold+1))
                    torch.save(best_param_score, 'best_param_score_{}_{}.pt'.format(config.expname ,fold+1))
                else:
                    i += 1

                
            model.load_state_dict(best_param_score)
            result += predict_result(model, test_loader)
            print('best_param_score_{}_{}.pt'.format(config.expname ,fold+1))
            torch.save(best_param_score, 'best_param_score_{}_{}.pt'.format(config.expname ,fold+1))
            
            result /= NUM_FOLDS
            
        del train_df, val_df, model, optimizer, criterion, scheduler
        torch.cuda.empty_cache()
        del valid_loader, train_loader, valid_set, train_set
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()