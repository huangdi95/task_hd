import os

class op_hd():
    def __init__(self):
        self.id = 0
        self.update_count = 0
        self.agg_count = 0
    def clean_id(self):
        self.id = 0
        self.update_count = 0
        self.agg_count = 0
    def VertexFeatureMultWeight(self, **args):
        args['task_name'] = '"aggregate_'+str(self.update_count)+'"'
        args['id'] = str(self.id) 
        out = \
'\n\
task {{\n\
  id: {id}\n\
  task_type: "VertexFeatureMultWeight"\n\
  task_name: {task_name}\n\
  vertex_feature_mult_weight_param{{\n\
    vertex_batch_size: {vertex_batch_size}\n\
    vertex_in_buf_max: {vertex_in_buf_max}\n\
    vertex_num: {vertex_num} #\n\
    vertex_feature_len: {vertex_feature_len} #\n\
    weight_input_num: {weight_input_num} #\n\
    weight_output_num: {weight_output_num} ##\n\
    weight_output_tiling_unit: {weight_output_tiling_unit}\n\
    fusing_active: {fusing_active} ##\n\
    active_func: {active_func} ##\n\
  }}\n\
  compt_efficiency: {compt_efficiency}\n\
  mem_efficiency: {mem_efficiency}\n\
  fusing_last_task: {fusing_last_task} ###\n\
  fusing_next_task: {fusing_next_task} ###\n\
  split_core: {split_core} ###\n\
  ep_core_num: {ep_core_num} ###\n\
  vp_core_num: {vp_core_num} ###\n\
}}\n\
    '.format(**args)
        self.update_count+=1
        self.id += 1
        return out
    
    def EdgeAggregation(self, **args):
        args['task_name'] = '"aggregate_'+str(self.agg_count)+'"'
        args['id'] = str(self.id) 
        out = \
'\n\
task {{\n\
  id: {id}\n\
  task_type: "EdgeAggregation"\n\
  task_name: {task_name}\n\
  edge_aggregation_param{{\n\
    edge_process_step: {edge_process_step}\n\
\n\
    edge_file: {edge_file} # 16:vertex_unit_len\n\
    edge_num: {edge_num} #\n\
    edge_unit_len: {edge_unit_len}\n\
    edge_read_step: {edge_read_step}\n\
    max_edge_in_buf: {max_edge_in_buf}\n\
\n\
    vertex_unit_len: {vertex_unit_len} ##\n\
    vertex_num: {vertex_num} #\n\
    max_vertex_in_buf: {max_vertex_in_buf}\n\
    max_agg_res_in_buf: {max_agg_res_in_buf}\n\
\n\
    agg_type: {agg_type} ##\n\
    fusing_active: {fusing_active} ##\n\
    active_func: {active_func} ##\n\
  }}\n\
  compt_efficiency: {compt_efficiency}\n\
  mem_efficiency: {mem_efficiency}\n\
  split_core: {split_core} ### \n\
  ep_core_num: {ep_core_num} ###\n\
  vp_core_num: {vp_core_num} ###\n\
  fusing_last_task: {fusing_last_task} ###\n\
  fusing_next_task: {fusing_next_task} ###\n\
}}\n\
    '.format(**args)
        self.agg_count+=1
        self.id += 1
        return out

