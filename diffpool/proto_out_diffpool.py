import os
from op_hd import op_hd

args={}
args['compt_efficiency'] = 0.90
args['mem_efficiency'] = 0.90
args['vertex_batch_size'] = 16
args['vertex_in_buf_max'] = 256
args['weight_output_tiling_unit'] = 16
args['edge_process_step'] = 256
args['edge_unit_len'] = 4
args['edge_read_step'] = 8
args['max_edge_in_buf'] = 1024
args['max_vertex_in_buf'] = 2048
args['max_agg_res_in_buf'] = 2048
args['dst_type'] = 'None'
args['id'] = 0
#args['task_type'] = "VertexFeatureMultWeight"
args['task_name'] = '"update_1"'

args['edge_file'] = '"bin_edges/edges.reddit.fea16.bin"' # 16 =vertex_unit_len
args['edge_num'] = 114848857 #
args['vertex_num'] = 232965 #
args['vertex_feature_len'] = 602 #
args['weight_input_num'] = 602 #

args['vertex_unit_len'] = 16 ##
args['agg_type'] ='WeightedSum' ##
args['weight_output_num'] = 16 ##
args['fusing_active'] = 0 ##
args['active_func'] = 'ReLU' ##

args['fusing_last_task'] = 0 ###
args['fusing_next_task'] = 0 ###
args['split_core'] = 0 ###
args['ep_core_num'] = 2 ###
args['vp_core_num'] = 2 ###

modes = ['TEST', 'TRAIN']
datasets = ['dd', 'enzymes']
vertexs = {'cora': 2708, 'citeseer': 3327, 'pubmed': 19717, 'ppi': 56944, 'reddit': 232965, 'dd': 334925, 'enzymes': 19580}
v_feature = {'cora': 1433, 'citeseer': 3703, 'pubmed': 500, 'ppi': 50, 'reddit': 602, 'dd': 89, 'enzymes': 18}
nclasses = {'cora': 7, 'citeseer': 6, 'pubmed': 3, 'ppi': 121, 'reddit': 41, 'dd': 574, 'enzymes': 12}
edges = {'cora': 10556, 'citeseer': 9228, 'pubmed': 88651, 'ppi': 889362, 'reddit': 114615892, 'dd': 1686092, 'enzymes': 74564}
corenames = ['unicore', 'dualcore']
cores = range(0, 4)

agg_types = ['Mean']
algo = 'diffpool'

def main():
    args['nd'] = 256
    args['kd'] = 1
    args['md'] = 256
    out_dim = 64 
    op = op_hd() 
    for mode in modes:
        for dataset in datasets:
            args['vertex_num'] = vertexs[dataset]#
            args['edge_num'] = edges[dataset]+vertexs[dataset]
            for agg_type in agg_types:
                for ep in cores:
                    op.clean_id()
                    is_split = 1 if ep else 0
                    args['ep_core_num'] = cores[ep] ###
                    args['vp_core_num'] = 4 - cores[ep] ###
                    if is_split:
                       description = '"'+algo+'.'+dataset+'.'+mode+'.'+corenames[is_split]+'_'+str(ep)+str(4-ep)+'"'
                    else:
                       description = '"'+algo+'.'+dataset+'.'+mode+'.'+corenames[is_split]+'"'
                    out = 'description: '+description+'\n'+'phase: '+mode+'\n'
                    last = 0
                    first = 1
                    
                    for layer in range(1):
                        if first:
                            args['weight_output_num'] = v_feature[dataset]
                        for i in range(5):
                            args['vertex_unit_len'] = v_feature[dataset] if first else out_dim ##
                            args['edge_file'] = '"bin_edges/edges.'+dataset+'.fea'+str(args['vertex_unit_len'])+'.bin"'
                            args['agg_type'] = agg_type ##
                            args['fusing_active'] = 0
                            args['active_func'] = 'Softmax' if last else 'ReLU'##
                            args['fusing_last_task'] = 0 ###
                            args['fusing_next_task'] = 1 ###
                            args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                            out += op.EdgeAggregation(**args)

                            args['vertex_feature_len'] = v_feature[dataset]*2 if first else args['weight_output_num']*2
                            args['weight_input_num'] = args['vertex_feature_len']
                            args['weight_output_num'] = nclasses[dataset] if last else out_dim 
                            args['fusing_active'] = 1
                            args['active_func'] = 'Softmax' if last else 'ReLU'##
                            args['fusing_last_task'] = 1 ###
                            args['fusing_next_task'] = 0 ###
                            args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                            out += op.VertexFeatureMultWeight(**args)
                            first = 0

                        args['vertex_unit_len'] = v_feature[dataset] if first else out_dim ##
                        args['edge_file'] = '"bin_edges/edges.'+dataset+'.fea'+str(args['vertex_unit_len'])+'.bin"'
                        args['agg_type'] = agg_type ##
                        args['fusing_active'] = 0
                        args['active_func'] = 'Softmax' if last else 'ReLU'##
                        args['fusing_last_task'] = 0 ###
                        args['fusing_next_task'] = 1 ###
                        args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                        out += op.EdgeAggregation(**args)

                        args['vertex_feature_len'] = v_feature[dataset]*2 if first else args['weight_output_num']*2
                        args['weight_input_num'] = args['vertex_feature_len']
                        args['weight_output_num'] = nclasses[dataset]
                        args['fusing_active'] = 0
                        args['active_func'] = 'Softmax' if last else 'ReLU'##
                        args['fusing_last_task'] = 1 ###
                        args['fusing_next_task'] = 0 ###
                        args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                        out += op.VertexFeatureMultWeight(**args)

                        args['n'] = nclasses[dataset]
                        args['k'] = args['vertex_num']
                        args['m'] = out_dim
                        args['m1_in_buf'] = 1
                        args['m2_in_buf'] = 1
                        args['out_in_buf'] = 1
                        args['fusing_last_task'] = 0 ###
                        args['fusing_next_task'] = 0 ###
                        args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                        out += op.MMM(**args)

                        args['n'] = nclasses[dataset]
                        args['k'] = args['vertex_num']
                        args['m'] = args['vertex_num']
                        args['m1_in_buf'] = 1
                        args['m2_in_buf'] = 0
                        args['out_in_buf'] = 1
                        args['fusing_last_task'] = 0 ###
                        args['fusing_next_task'] = 0 ###
                        args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                        out += op.MMM(**args)

                        args['n'] = nclasses[dataset]
                        args['k'] = args['vertex_num']
                        args['m'] = nclasses[dataset]
                        args['m1_in_buf'] = 1
                        args['m2_in_buf'] = 1
                        args['out_in_buf'] = 1
                        args['fusing_last_task'] = 0 ###
                        args['fusing_next_task'] = 0 ###
                        args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                        out += op.MMM(**args)

                        for i in range(3):
                            args['n'] = nclasses[dataset]
                            args['k'] = nclasses[dataset]
                            args['m'] = out_dim
                            args['m1_in_buf'] = 1
                            args['m2_in_buf'] = 1
                            args['out_in_buf'] = 1
                            args['fusing_last_task'] = 0 ###
                            args['fusing_next_task'] = 0 ###
                            args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                            out += op.MMM(**args)

                            args['n'] = nclasses[dataset]
                            args['k'] = out_dim
                            args['m'] = out_dim
                            args['m1_in_buf'] = 1
                            args['m2_in_buf'] = 0
                            args['out_in_buf'] = 0 if i == 2 else 1
                            args['fusing_last_task'] = 0 ###
                            args['fusing_next_task'] = 0 ###
                            args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                            out += op.MMM(**args)

                    name = description.replace('"','')+'.pt'
                    with open(name,'w') as file_obj:
                        file_obj.write(out)
                #        doc = open('out.txt','w')
                #        print(data_dict,file=doc)


if __name__ == '__main__':
    main()
