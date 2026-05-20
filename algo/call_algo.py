# import all algorithms this benchmark implement

def call_algo(algo_name, config, device):
    
    algo_name = algo_name.lower()
    assert algo_name in ['dara', 'bosa', 'iql', 'td3_bc', 'igdf', "root", "otdf"]
    # offline offline setting
    from offline_offline.dara import DARA
    from offline_offline.bosa import BOSA
    from offline_offline.iql import IQL
    from offline_offline.td3_bc import TD3BC
    from offline_offline.igdf import IGDF
    from offline_offline.otdf import OTDF
    from offline_offline.root import ROOT

    algo_to_call = {
        'dara': DARA,
        'bosa': BOSA,
        'iql': IQL,
        'td3_bc': TD3BC,
        'igdf': IGDF,
        'otdf': OTDF,
        'root': ROOT,
    }

    algo = algo_to_call[algo_name]
    policy = algo(config, device)
    
    return policy