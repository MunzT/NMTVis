import data as d
import os
import random
import seq2seq.hp as hp
import shared
import time
import torch
import torch.nn as nn
from finetuningdataset import FinetuneDataset
from seq2seq.hp import teacher_forcing_ratio, clip, batch_size, learning_rate
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import BucketIterator

use_cuda = torch.cuda.is_available()


# Pad a with the PAD symbol# Pad a
def pad_seq(seq, max_length, vocab):
    seq += [vocab.stoi[d.BLANK_WORD] for i in range(max_length - len(seq))]
    return seq


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = learning_rate * (0.1 ** (epoch // 10))
    print("Adjusted learning rate to {}".format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def adjust_teacher_forcing(iter):
    return teacher_forcing_ratio * (0.9 ** ((iter - 1) // 20000))


def train(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, batch_size=batch_size,
          teacher_forcing_ratio=teacher_forcing_ratio, device=torch.device('cpu')):

    encoder.train()
    decoder.train()

    if encoder_optimizer:
        encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    loss = 0.0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_hidden = encoder_hidden
    last_attn_vector = torch.zeros((batch_size, decoder.hidden_size), device=device)

    max_target_length = min(max(target_lengths.tolist()), d.MAX_LEN)

    teacher_force = random.random() < teacher_forcing_ratio
    acc = []
    if teacher_force:
        # Run through decoder one time step at a time
        for t in range(max_target_length-1):
            decoder_input = target_batches[:,t]  # Next input is current target
            decoder_output, decoder_hidden, _, last_attn_vector = decoder(
                decoder_input, decoder_hidden, encoder_outputs, last_attn_vector
            )
            loss += criterion(decoder_output, target_batches[:,t+1])
            acc.append(decoder_output.argmax(dim=-1).eq(target_batches[:, t+1]).float().view(-1).mean(dim=-1).item())
    else:
        # Run through decoder one time step at a time
        decoder_input = torch.tensor([d.TGT.vocab.stoi[d.BOS_WORD]] * batch_size, dtype=torch.long, device=device)
        for t in range(max_target_length-1):
            decoder_output, decoder_hidden, _, last_attn_vector = decoder(
                decoder_input, decoder_hidden, encoder_outputs, last_attn_vector
            )

            _, i = decoder_output.topk(1)

            loss += criterion(decoder_output, target_batches[:,t+1])
            acc.append(decoder_output.argmax(dim=-1).eq(target_batches[:, t+1]).float().view(-1).mean(dim=-1).item())
            decoder_input = i.view(-1).detach()
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    if encoder_optimizer:
        encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / max_target_length, sum(acc) / len(acc)


def train_iters(seq2seq_model, train_data, eval_data, SRC_LANG, TGT_LANG,
                n_epochs=hp.n_epochs,
                print_every=hp.print_loss_every_iters,
                evaluate_every=hp.eval_bleu_every_epochs,
                save_every=hp.save_every_epochs,
                learning_rate=hp.learning_rate,
                decoder_learning_ratio=hp.decoder_learning_ratio,
                batch_size=hp.batch_size,
                encoder_optimizer_state=None,
                decoder_optimizer_state=None, train_loss=[], eval_loss=[],
                bleu_scores=[],
                start_epoch=1,
                retrain=False,
                weight_decay=1e-5,
                device=torch.device('cpu'),
                writer: SummaryWriter = None, save_model = True):
    encoder = seq2seq_model.encoder
    decoder = seq2seq_model.decoder

    encoder.train()
    decoder.train()

    optim_type = optim.Adam if not retrain else optim.Adagrad

    # Initialize optimizers and criterion
    if not retrain or retrain:
        encoder_optimizer = optim_type(filter(lambda p: p.requires_grad, encoder.parameters()), lr=learning_rate,
                                       weight_decay=weight_decay)
    else:
        encoder_optimizer = None

    decoder_optimizer = optim_type(filter(lambda p: p.requires_grad, decoder.parameters()),
                                   lr=learning_rate * decoder_learning_ratio, weight_decay=weight_decay)

    if encoder_optimizer_state:
        encoder_optimizer.load_state_dict(encoder_optimizer_state)
    if decoder_optimizer_state:
        decoder_optimizer.load_state_dict(decoder_optimizer_state)

    criterion = nn.CrossEntropyLoss(ignore_index=d.TGT.vocab.stoi[d.BLANK_WORD])
    criterion.to(device)


    train_iter = BucketIterator(train_data, batch_size,
                            sort_key=lambda x: (len(x.src), len(x.trg)),
                            train=True, shuffle=True, sort_within_batch=True)
    if eval_data:
        val_iter = BucketIterator(eval_data, batch_size, 
                                    sort_key=lambda x: (len(x.src), len(x.trg)),
                                    train=False, sort_within_batch=True)

    print("Starting training for n_epochs={} lr={}, batch_size={}".format(n_epochs, learning_rate, batch_size))
    for epoch in range(start_epoch, n_epochs + 1):
        # lr = adjust_learning_rate(decoder_optimizer, epoch)
        epoch_training_loss = []
        epoch_training_accs = []

        train_batches = iter(train_iter)
        for i, batch in enumerate(train_batches):
            teacher_forcing_ratio = adjust_teacher_forcing(i)

            # Get training data for this cycle
            src = batch.src
            src_lengths = (src != d.SRC.vocab.stoi[d.BLANK_WORD]).sum(dim=1)
            src = src.to(device)
            batch_size = src.size(dim=0)
            trg = batch.trg
            tgt_lengths = (trg != d.TGT.vocab.stoi[d.BLANK_WORD]).sum(dim=1)
            trg = trg.to(device)

            # Run the train function
            loss, train_acc = train(
                src, src_lengths, trg, tgt_lengths,
                encoder, decoder,
                encoder_optimizer, decoder_optimizer, criterion, batch_size=batch_size,
                teacher_forcing_ratio=teacher_forcing_ratio,
                device=device
            )

            # Keep track of loss
            epoch_training_loss.append(loss)
            epoch_training_accs.append(train_acc)


        # Log epoch
        avg_training_loss = sum(epoch_training_loss) / len(epoch_training_loss)
        avg_training_accs = sum(epoch_training_accs) / len(epoch_training_accs)
        if writer:
            writer.add_scalar("train/loss", avg_training_loss, epoch)
            writer.add_scalar("train/acc", avg_training_accs, epoch)

        # evaluate
        if retrain:
            continue

        encoder.eval()
        decoder.eval()

        epoch_evaluation_losses = []
        epoch_evaluation_accs = []
        val_batches = iter(val_iter)
        for i, batch in enumerate(val_batches):
            src = batch.src
            src_lengths = (src != d.SRC.vocab.stoi[d.BLANK_WORD]).sum(dim=1)
            src = src.to(device)
            batch_size = src.size(dim=0)
            trg = batch.trg
            tgt_lengths = (trg != d.TGT.vocab.stoi[d.BLANK_WORD]).sum(dim=1)
            trg = trg.to(device)

            curr_eval_loss, curr_eval_acc = eval(src, src_lengths, trg, tgt_lengths,
                                  encoder, decoder, criterion, batch_size, device=device)
            epoch_evaluation_losses.append(curr_eval_loss)
            epoch_evaluation_accs.append(curr_eval_acc)

        avg_evaluation_loss = sum(epoch_evaluation_losses) / len(epoch_evaluation_losses)
        avg_evaluation_acc = sum(epoch_evaluation_accs) / len(epoch_evaluation_accs)
        if epoch % evaluate_every == 0:
            src_file = f'.data/wmt14/newstest2013.tok.bpe.32000.{SRC_LANG}'
            tgt_file = f'.data/wmt14/newstest2013.tok.bpe.32000.{TGT_LANG}'
            bleu = seq2seq_model.eval_bleu(src_file, tgt_file)
            print("BLEU = {}".format(bleu))
            bleu_scores.append(bleu)
            if writer:
                writer.add_scalar('eval/bleu', bleu, epoch)

        train_loss.append(avg_training_loss)
        eval_loss.append(avg_evaluation_loss)

        if writer:
            writer.add_scalar("eval/loss", avg_evaluation_loss, epoch)
            writer.add_scalar("eval/acc", avg_evaluation_acc, epoch)

        if epoch % save_every == 0 and save_model == True:
            torch.save({
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "eval_loss": eval_loss,
                'eval_acc': avg_evaluation_acc,
                "bleu_scores": bleu_scores,
                "max_length": 50,
                "min_length": hp.MIN_LENGTH
            }, f".data/models/seq2seq/seq2seq_{SRC_LANG}_{TGT_LANG}_epoch_{epoch}.pt")

        print("Avg. Training Loss: %.2f Avg. Evaluation Loss: %.2f" % (avg_training_loss, avg_evaluation_loss)) 

    return encoder_optimizer.state_dict() if encoder_optimizer else None, decoder_optimizer.state_dict()


@torch.no_grad()
def eval(input_batches, input_lengths, target_batches, target_lengths, encoder, decoder, criterion, batch_size, device="cpu"):
    # Zero gradients of both optimizers
    encoder.eval()
    decoder.eval()
    loss = 0  # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = torch.tensor([d.TGT.vocab.stoi[d.BOS_WORD]] * batch_size, dtype=torch.long, device=device)
    decoder_hidden = encoder_hidden
    last_attn_vector = torch.zeros((batch_size, decoder.hidden_size), device=device)

    max_target_length = min(max(target_lengths.tolist()), d.MAX_LEN)

    # Run through decoder one time step at a time
    acc = []
    for t in range(max_target_length-1):
        decoder_output, decoder_hidden, _, last_attn_vector = decoder(
            decoder_input, decoder_hidden, encoder_outputs, last_attn_vector
        )
        loss += criterion(decoder_output, target_batches[:, t+1])
        acc.append(decoder_output.argmax(dim=-1).eq(target_batches[:, t+1]).float().view(-1).mean(dim=-1).item())
        decoder_input = target_batches[:, t+1]  # Next input is current target

    encoder.train()
    decoder.train()
    return loss.item() / max_target_length, sum(acc) / len(acc)


def retrain_iters(seq2seq_model, pairs, eval_pairs,
                  src_lang, tgt_lang,
                  n_epochs=hp.n_epochs,
                  print_every=hp.print_loss_every_iters,
                  evaluate_every=hp.eval_bleu_every_epochs,
                  save_every=hp.save_every_epochs,
                  learning_rate=hp.learning_rate,
                  decoder_learning_ratio=hp.decoder_learning_ratio,
                  batch_size=hp.batch_size,
                  weight_decay=1e-5,
                  encoder_optimizer_state=None,
                  decoder_optimizer_state=None,
                  device=torch.device('cpu'),
                  save_ckpt=True):
    encoder, decoder = seq2seq_model.encoder, seq2seq_model.decoder

    print("Training for", n_epochs, 'epochs')

    for param in encoder.parameters():
        param.requires_grad = False

    for param in encoder.lstm.parameters():
        param.requires_grad = True

    train_set = FinetuneDataset(pairs=pairs, fields=(d.SRC, d.TGT))
    enc_state, dec_state = train_iters(seq2seq_model, train_set, eval_pairs, src_lang, tgt_lang, n_epochs, print_every, evaluate_every,
                       save_every, learning_rate, decoder_learning_ratio, batch_size,
                       encoder_optimizer_state=encoder_optimizer_state,
                       decoder_optimizer_state=decoder_optimizer_state,
                       start_epoch=1, retrain=True, device=device, save_model=False)

    if save_ckpt:
        filename = os.path.join(shared.SEQ2SEQ_CHECKPOINT_PATH, f"seq2seq_{src_lang}_{tgt_lang}_{time.strftime('%Y%m%d-%H%M%S')}_finetuned.pt")
        print('saving to...', filename)
        torch.save({
                    "encoder": enc_state,
                    "decoder": dec_state,
                    "encoder_optimizer": encoder_optimizer_state,
                    "decoder_optimizer": decoder_optimizer_state,
                    "epoch": n_epochs,
                    "max_length": 40,
                    "min_length": hp.MIN_LENGTH
                }, filename)

    return enc_state, dec_state


