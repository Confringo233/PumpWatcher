import argparse
import traceback
import time
import copy

import torch.nn.functional as F
from imblearn.over_sampling import SMOTE
import numpy as np
import dgl
import torch
import lightgbm as lgb
from torch import nn

from tgn import TGN
from data_preprocess import TemporalPDDataset
from dataloading import (
                            FastTemporalEdgeCollator, FastTemporalSampler,
                            SimpleTemporalEdgeCollator, SimpleTemporalSampler,
                            TemporalEdgeCollator, TemporalSampler,
                            TemporalEdgeDataLoader
                        )

TRAIN_SPLIT = 0.7
VALID_SPLIT = 0.85

# set random Seed
np.random.seed(2024)
torch.manual_seed(2024)
torch.cuda.manual_seed_all(2024)


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, features, labels):
        positive_pairs = features[labels == 1]
        negative_pairs = features[labels == 0]

        same_loss1 = torch.tensor(0.0, device=features.device)
        same_loss2 = torch.tensor(0.0, device=features.device)
        diff_loss = torch.tensor(0.0, device=features.device)

        if positive_pairs.size(0) > 0:
            same_distances1 = torch.cdist(positive_pairs, positive_pairs, p=2)
            num_same1 = same_distances1.size(0)
            same_loss1 = same_distances1.mean()
        else:
            num_same1 = 0

        if negative_pairs.size(0) > 0:
            same_distances2 = torch.cdist(negative_pairs, negative_pairs, p=2)
            num_same2 = same_distances2.size(0)
            same_loss2 = same_distances2.mean()
        else:
            num_same2 = 0

        if positive_pairs.size(0) > 0 and negative_pairs.size(0) > 0:
            diff_distances = torch.cdist(positive_pairs, negative_pairs, p=2)
            num_diff = diff_distances.size(0)
            diff_loss = F.relu(diff_distances - self.margin).mean()
        else:
            num_diff = 0

        total_samples = num_same1 + num_same2 + num_diff
        if total_samples > 0:
            same_weight = num_diff / total_samples
            diff_weight = (num_same1 + num_same2) / total_samples
        else:
            same_weight = 0.5
            diff_weight = 0.5

        total_loss = (same_loss1 + same_loss2) * same_weight + diff_loss * diff_weight

        return total_loss




def judgement(y_pred_list, y_true_list):
    FP = sum((y_true_list[i] == 0) and (y_pred_list[i] == 1) for i in range(len(y_pred_list)))
    TN = sum((y_true_list[i] == 0) and (y_pred_list[i] == 0) for i in range(len(y_pred_list)))
    FN = sum((y_true_list[i] == 1) and (y_pred_list[i] == 0) for i in range(len(y_pred_list)))
    TP = sum((y_true_list[i] == 1) and (y_pred_list[i] == 1) for i in range(len(y_pred_list)))

    FNR = FN / (TP + FN) if (TP + FN) != 0 else 0       # FNR
    FPR = FP / (FP + TN) if (FP + TN) != 0 else 0       # FPR
    Precision = TP / (TP + FP) if (TP + FP) != 0 else 0     # Precision
    Recall = TP / (TP + FN) if (TP + FN) != 0 else 0        # Recall
    F1 = 2 * Precision * Recall / (Precision + Recall) if (Precision + Recall) != 0 else 0      # F1 score
    TPR = TP / (TP + FN) if (TP + FN) != 0 else 0       # TPR
    TNR = TN / (TN + FP) if (TN + FP) != 0 else 0       # TNR
    BAC = (TPR + TNR) / 2       # BAC
    # print(' TP = {}, FP = {}, TN = {}, FN = {} \n FPR = {} FNR = {} F1 = {} BAC = {}'.format(TP, FP, TN, FN, FPR, FNR, F1, BAC))
    return FPR, FNR, F1, BAC




def train(model, dataloader, sampler, criterion, optimizer, args, device, epoch):
    model.train()
    batch_cnt = 0
    train_loss = 0
    edge_adv_embed_list = []
    true_label_list = []
    last_t = time.time()
    for _, g, _, blocks in dataloader:
        g = g.to(device)
        optimizer.zero_grad()
        edge_embed = model.get_edge_embedding(g, blocks)         # 先生成边embedding
        edge_adv_embed = model.get_edge_feature(g)
        true_label = g.edata['label']
        loss = criterion(edge_adv_embed, true_label)
        train_loss += loss.item()
        retain_graph = True if batch_cnt == 0 and not args.fast_mode else False
        loss.backward(retain_graph=retain_graph)
        optimizer.step()
        model.memory.to(device)
        model.detach_memory()
        if not args.not_use_memory:
            model.update_memory(g)
        if args.fast_mode:
            sampler.attach_last_update(model.memory.last_update_t)
        print("Batch:{}-{}, Loss: {}, Time: {}".format(epoch, batch_cnt, loss, time.time() - last_t))
        last_t = time.time()
        batch_cnt += 1

        edge_adv_embed_list += edge_adv_embed.tolist()
        true_label_list += true_label.squeeze().tolist()

    return edge_adv_embed_list, true_label_list, train_loss / len(dataloader)


def test(model, dataloader, sampler, criterion, args, device):
    model.eval()
    edge_feature_list = []
    true_label_list = []
    val_loss = 0
    model.memory.reset_memory()
    with torch.no_grad():
        for _, g, _, blocks in dataloader:
            g = g.to(device)

            model.get_edge_embedding(g, blocks)
            edge_adv_embed = model.get_edge_feature(g)
            true_label = g.edata['label']
            loss = criterion(edge_adv_embed, true_label)
            val_loss += loss.item()
            model.memory.to(device)
            if not args.not_use_memory:
                model.update_memory(g)
            if args.fast_mode:
                sampler.attach_last_update(model.memory.last_update_t)

            edge_feature_list += edge_adv_embed.tolist()
            true_label_list += true_label.squeeze().tolist()

    return edge_feature_list, true_label_list, val_loss / len(dataloader)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=30,
                        help='epochs for training on entire dataset')
    parser.add_argument("--batch_size", type=int,
                        default=512, help="Size of each batch")
    parser.add_argument("--embedding_dim", type=int, default=16,
                        help="Embedding dim for link prediction")
    parser.add_argument("--memory_dim", type=int, default=16,
                        help="dimension of memory")
    parser.add_argument("--temporal_dim", type=int, default=16,
                        help="Temporal dimension for time encoding")
    parser.add_argument("--memory_updater", type=str, default='gru',
                        help="Recurrent unit for memory update")
    parser.add_argument("--aggregator", type=str, default='last',
                        help="Aggregation method for memory update")
    parser.add_argument("--n_neighbors", type=int, default=10,
                        help="number of neighbors while doing embedding")
    parser.add_argument("--sampling_method", type=str, default='topk',
                        help="In embedding how node aggregate from its neighor")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of heads for multihead attention mechanism")
    parser.add_argument("--fast_mode", action="store_true", default=False,
                        help="Fast Mode uses batch temporal sampling, history within same batch cannot be obtained")
    parser.add_argument("--simple_mode", action="store_true", default=True,
                        help="Simple Mode directly delete the temporal edges from the original static graph")
    parser.add_argument("--num_negative_samples", type=int, default=0,
                        help="number of negative samplers per positive samples")
    parser.add_argument("--dataset", type=str, default="pd")
    parser.add_argument("--k_hop", type=int, default=2,
                        help="sampling k-hop neighborhood")
    parser.add_argument("--not_use_memory", action="store_true", default=False,
                        help="Enable memory for TGN Model disable memory for TGN Model")


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()

    assert not (args.fast_mode and args.simple_mode), "you can only choose one sampling mode"

    if args.k_hop != 1:
        assert args.simple_mode, "this k-hop parameter only support simple mode"

    data = TemporalPDDataset()


    # Pre-process data, mask new node in test set from original graph
    num_nodes = data.num_nodes()
    num_edges = data.num_edges()

    trainval_div = int(VALID_SPLIT * num_edges)

    # Select new node from test set and remove them from entire graph
    test_split_ts = data.edata['timestamp'][trainval_div]
    test_nodes = torch.cat([data.edges()[0][trainval_div:], data.edges()[1][trainval_div:]]).unique().cpu().numpy()
    test_new_nodes = np.random.choice(test_nodes, int(0*len(test_nodes)), replace=False)

    in_subg = dgl.in_subgraph(data, test_new_nodes)
    out_subg = dgl.out_subgraph(data, test_new_nodes)
    # Remove edge who happen before the test set to prevent from learning the connection info
    new_node_in_eid_delete = in_subg.edata[dgl.EID][in_subg.edata['timestamp'] < test_split_ts]
    new_node_out_eid_delete = out_subg.edata[dgl.EID][out_subg.edata['timestamp'] < test_split_ts]
    new_node_eid_delete = torch.cat([new_node_in_eid_delete, new_node_out_eid_delete]).unique()

    graph_new_node = copy.deepcopy(data)
    # relative order preseved
    graph_new_node.remove_edges(new_node_eid_delete)

    # Now for no new node graph, all edge id need to be removed
    in_eid_delete = in_subg.edata[dgl.EID]
    out_eid_delete = out_subg.edata[dgl.EID]
    eid_delete = torch.cat([in_eid_delete, out_eid_delete]).unique()

    graph_no_new_node = copy.deepcopy(data)
    graph_no_new_node = graph_no_new_node
    graph_no_new_node.remove_edges(eid_delete)

    # graph_no_new_node and graph_new_node should have same set of nid

    # Sampler Initialization
    if args.simple_mode:
        fan_out = [args.n_neighbors for _ in range(args.k_hop)]
        sampler = SimpleTemporalSampler(graph_no_new_node, fan_out)
        new_node_sampler = SimpleTemporalSampler(data, fan_out)
        edge_collator = SimpleTemporalEdgeCollator
    elif args.fast_mode:
        sampler = FastTemporalSampler(graph_no_new_node, k=args.n_neighbors)
        new_node_sampler = FastTemporalSampler(data, k=args.n_neighbors)
        edge_collator = FastTemporalEdgeCollator
    else:
        sampler = TemporalSampler(k=args.n_neighbors)
        edge_collator = TemporalEdgeCollator

    neg_sampler = dgl.dataloading.negative_sampler.Uniform(k=args.num_negative_samples)
    # Set Train, validation, test and new node test id
    train_seed = torch.arange(int(TRAIN_SPLIT * graph_no_new_node.num_edges()))
    valid_seed = torch.arange(int(TRAIN_SPLIT * graph_no_new_node.num_edges()),trainval_div - new_node_eid_delete.size(0))
    test_seed = torch.arange(trainval_div - new_node_eid_delete.size(0), graph_no_new_node.num_edges())
    test_new_node_seed = torch.arange(trainval_div - new_node_eid_delete.size(0), graph_new_node.num_edges())

    g_sampling = None if args.fast_mode else dgl.add_reverse_edges(graph_no_new_node, copy_edata=True)
    new_node_g_sampling = None if args.fast_mode else dgl.add_reverse_edges(graph_new_node, copy_edata=True)
    if not args.fast_mode:
        new_node_g_sampling.ndata[dgl.NID] = new_node_g_sampling.nodes()
        g_sampling.ndata[dgl.NID] = new_node_g_sampling.nodes()

    # we highly recommend that you always set the num_workers=0, otherwise the sampled subgraph may not be correct.

    # train_seed = train_seed.to(device)
    train_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                              train_seed,
                                              sampler,
                                              batch_size=args.batch_size,
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=g_sampling,
                                              device=device)


    valid_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                              valid_seed,
                                              sampler,
                                              batch_size=args.batch_size,
                                              negative_sampler=neg_sampler,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=0,
                                              collator=edge_collator,
                                              g_sampling=g_sampling,
                                              device=device)

    test_dataloader = TemporalEdgeDataLoader(graph_no_new_node,
                                             test_seed,
                                             sampler,
                                             batch_size=args.batch_size,
                                             negative_sampler=neg_sampler,
                                             shuffle=False,
                                             drop_last=False,
                                             num_workers=0,
                                             collator=edge_collator,
                                             g_sampling=g_sampling,
                                             device=device)

    test_new_node_dataloader = TemporalEdgeDataLoader(graph_new_node,
                                                      test_new_node_seed,
                                                      new_node_sampler if args.fast_mode else sampler,
                                                      batch_size=args.batch_size,
                                                      negative_sampler=neg_sampler,
                                                      shuffle=False,
                                                      drop_last=False,
                                                      num_workers=0,
                                                      collator=edge_collator,
                                                      g_sampling=new_node_g_sampling,
                                                      device=device)

    edge_dim = data.edata['feats'].shape[1]
    num_node = data.num_nodes()

    model = TGN(edge_feat_dim=edge_dim,
                memory_dim=args.memory_dim,
                temporal_dim=args.temporal_dim,
                embedding_dim=args.embedding_dim,
                num_heads=args.num_heads,
                num_nodes=num_node,
                n_neighbors=args.n_neighbors,
                memory_updater_type=args.memory_updater,
                layers=args.k_hop)
    model = model.to(device)

    criterion = ContrastiveLoss(margin=0.5)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

    params = {
        'objective': 'binary',      # 二分类任务
        'metric': 'binary_error',   # 评估指标为错误率
        'num_leaves': 127,          # 最大叶子节点数
        'learning_rate': 0.1,       # 学习率
        # 'feature_fraction': 0.9,    # 训练时每次迭代选择特征的比例
        # 'bagging_fraction': 0.9,    # 训练时每次迭代选择数据的比例
        'bagging_freq': 5,          # 每次bagging的迭代次数
        'verbose': 0,   # 不输出训练过程中的信息
        'seed': 42      # 设置随机种子
    }

    train_edge_embed_list = []
    train_true_label_list = []
    train_loss_array = []

    best_val_loss = float('inf')
    best_model_path = './model/best_model.pth'

    # Implement Logging mechanism
    f = open("logging.txt", 'w')
    log_content = []
    if args.fast_mode:
        sampler.reset()
    try:
        for i in range(args.epochs):
            train_edge_embed_list, train_true_label_list, train_loss = train(model, train_dataloader, sampler, criterion, optimizer, args, device, i)
            train_loss_array.append(train_loss)
            memory_checkpoint = model.store_memory()
            if args.fast_mode:
                new_node_sampler.sync(sampler)
            model.restore_memory(memory_checkpoint)
            sample_nn = new_node_sampler if args.fast_mode else sampler
            if i < args.epochs - 1 and args.fast_mode:
                sampler.reset()
            model.reset_memory()
            valid_edge_embed_list, valid_true_label_list, valid_loss = test(model, valid_dataloader, sampler, criterion, args, device)

            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                torch.save(model.state_dict(), best_model_path)
                best_train_edge_embed_list = train_edge_embed_list
                best_train_true_label_list = train_true_label_list
                best_valid_edge_embed_list = valid_edge_embed_list
                best_valid_true_label_list = valid_true_label_list


        model.load_state_dict(torch.load(best_model_path))
        test_edge_embed_list, test_true_label_list, _ = test(model, test_dataloader, sampler, criterion, args, device)

        oversampler = SMOTE(random_state=42)

        train_edge_embed = np.array(best_train_edge_embed_list)
        train_true_label = np.array(best_train_true_label_list)

        X_resampled, y_resampled = oversampler.fit_resample(train_edge_embed, train_true_label)

        valid_edge_embed = np.array(best_valid_edge_embed_list)
        valid_true_label = np.array(best_valid_true_label_list)

        test_edge_embed = np.array(test_edge_embed_list)
        test_true_label = np.array(test_true_label_list)

        train_data = lgb.Dataset(X_resampled, label=y_resampled)
        val_data = lgb.Dataset(valid_edge_embed, label=valid_true_label, reference=train_data)

        bst = lgb.train(params, train_data, num_boost_round=100, valid_sets=[val_data])

        y_pred_valid = bst.predict(valid_edge_embed)
        y_pred_valid_class = np.round(y_pred_valid)
        valid_FPR, valid_FNR, valid_F1, valid_BAC = judgement(y_pred_valid_class, valid_true_label_list)

        y_pred_test = bst.predict(test_edge_embed)
        y_pred_test_class = np.round(y_pred_test)
        test_FPR, test_FNR, test_F1, test_BAC = judgement(y_pred_test_class, test_true_label_list)

        # log_content.append("Validation FPR = {:.3f} FNR = {:.3f} F1 = {:.3f} BAC = {:.3f}\n".format(valid_FPR,  valid_FNR,  valid_F1,  valid_BAC))
        log_content.append("Test FPR = {:.3f} FNR = {:.3f} F1 = {:.3f} BAC = {:.3f}\n".format(test_FPR, test_FNR, test_F1, test_BAC))

        for info in log_content:
            print(info)
        f.writelines(log_content)

        # with open('./output/train_loss_array.csv', 'w') as file:
        #     for element in train_loss_array:
        #         file.write(f"{element}\n")
        # with open('./output/test_edge_embed.csv', 'w') as file:
        #     for row in test_edge_embed:
        #         row_str = ' '.join(map(str, row.tolist()))
        #         file.write(row_str + '\n')
        # with open('./output/y_pred_test.csv', 'w') as file:
        #     for element in y_pred_test:
        #         file.write(f"{element}\n")
        # with open('./output/y_pred_test_class.csv', 'w') as file:
        #     for element in y_pred_test_class:
        #         file.write(f"{element}\n")
        # with open('./output/test_true_label.csv', 'w') as file:
        #     for element in test_true_label:
        #         file.write(f"{element}\n")

    except KeyboardInterrupt:
        traceback.print_exc()
        error_content = "Training Interreputed!"
        f.writelines(error_content)
        f.close()
    print("========Training is Done========")
