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
datasets = ['babi']
vertexs = {'cora': 2708, 'citeseer': 3327, 'pubmed': 19717, 'ppi': 56944, 'reddit': 232965, 'babi': 8}
v_feature = {'cora': 1433, 'citeseer': 3703, 'pubmed': 500, 'ppi': 50, 'reddit': 602, 'babi': 40}
nclasses = {'cora': 7, 'citeseer': 6, 'pubmed': 3, 'ppi': 121, 'reddit': 41}
edges = {'cora': 10556, 'citeseer': 9228, 'pubmed': 88651, 'ppi': 889362, 'reddit': 114615892, 'babi': 15}
corenames = ['unicore', 'dualcore']
cores = range(0, 4)

agg_types = ['WeightedSum']
algo = 'ggnn'

def main():
    hidden_dim = 40 
    edgetypes = 2
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
                    
                    for layer in range(2):
                        for i in range(4):
                            args['vertex_feature_len'] = v_feature[dataset] if first else hidden_dim
                            args['weight_input_num'] = v_feature[dataset] if first else hidden_dim
                            args['weight_output_num'] = hidden_dim
                            args['fusing_active'] = 0
                            args['active_func'] = 'Softmax' if last else 'ReLU'##
                            args['fusing_last_task'] = 0 ###
                            args['fusing_next_task'] = 0 ###
                            args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                            out += op.VertexFeatureMultWeight(**args)
                        first = 0

                        args['vertex_unit_len'] = args['weight_output_num'] ##
                        args['edge_file'] = '"bin_edges/edges.'+dataset+'.fea'+str(args['vertex_unit_len'])+'.bin"'
                        args['agg_type'] = agg_type ##
                        args['fusing_active'] = 0
                        args['active_func'] = 'Softmax' if last else 'ReLU'##
                        args['fusing_last_task'] = 0 ###
                        args['fusing_next_task'] = 1 ###
                        args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                        out += op.EdgeAggregation(**args)

                        args['vertex_feature_len'] = v_feature[dataset] if first else args['weight_output_num']
                        args['weight_input_num'] = v_feature[dataset] if first else args['weight_output_num']
                        args['weight_output_num'] = nclasses[dataset] if last else out_dim 
                        args['fusing_active'] = 1
                        args['active_func'] = 'Softmax' if last else 'ReLU'##
                        args['fusing_last_task'] = 1 ###
                        args['fusing_next_task'] = 0 ###
                        args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                        out += op.VertexFeatureMultWeight(**args)

                        args['vertex_feature_len'] = v_feature[dataset] if layer==0 else out_dim*2
                        args['weight_input_num'] = v_feature[dataset] if layer==0 else out_dim*2
                        args['weight_output_num'] = nclasses[dataset] if last else out_dim 
                        args['fusing_active'] = 1
                        args['active_func'] = 'Softmax' if last else 'ReLU'##
                        args['fusing_last_task'] = 0 ###
                        args['fusing_next_task'] = 0 ###
                        args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                        out += op.VertexFeatureMultWeight(**args)

                    last = 1
                    args['vertex_feature_len'] = v_feature[dataset] if first else out_dim*2
                    args['weight_input_num'] = v_feature[dataset] if first else out_dim*2
                    args['weight_output_num'] = nclasses[dataset] if last else out_dim 
                    args['fusing_active'] = 1
                    args['active_func'] = 'Softmax' if last else 'ReLU'##
                    args['fusing_last_task'] = 0 ###
                    args['fusing_next_task'] = 0 ###
                    args['split_core'] = is_split & (args['fusing_last_task'] | args['fusing_next_task'])###
                    out += op.VertexFeatureMultWeight(**args)

                    name = description.replace('"','')+'.pt'
                    with open(name,'w') as file_obj:
                        file_obj.write(out)
                #        doc = open('out.txt','w')
                #        print(data_dict,file=doc)


if __name__ == '__main__':
    main()
