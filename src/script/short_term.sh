# short-term prediction
python main.py --device_id 2 --machine machine  --t_patch_size 4  --his_len 12 --pred_len 12  --dataset GraphSH_28_96*FlowSH_7_96*GraphBJ_28_96*TaxiBJ13_48*GraphNJ_28_96*TaxiNYCIn_48*TaxiNYCOut_48*CrowdBJ_24*PopSH_24*CrowdNJ_24  --used_data 'GridGraphall'   --num_memory 512   --prompt_content 'node_graph'   --size middle --batch_ratio 0.4
