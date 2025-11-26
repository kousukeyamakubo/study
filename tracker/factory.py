# factory.py
from config import Config
from interfaces.tracker import ITracker
from interfaces.filter import IFilter

# 具体的な実装クラスをここでインポート
# 将来新しいアルゴリズムを追加するときは、ここにimportを追加するだけでOK
from core.kalman_filter import LinearKalmanFilter
from core.extended_kalman_filter import ExtendedKalmanFilter
from modules.gatings.ellipsoidal import EllipsoidalGating
from modules.associations.jpda import JPDAAssociation
from modules.associations.gnn import GNNAssociation
from modules.trackers.standard_tracker import StandardTracker

def create_filter(cfg: Config, F=None, H=None, Q=None, R=None) -> IFilter:
    """フィルタの生成（KF / EKF）"""
    if cfg.filter_type == "KF":
        return LinearKalmanFilter(F, H, Q, R)
    elif cfg.filter_type == "EKF":
        return ExtendedKalmanFilter(cfg) # EKFはConfigや関数を受け取る想定
    else:
        raise ValueError(f"Unknown filter type: {cfg.filter_type}")

def create_tracker(cfg: Config, filter: IFilter) -> ITracker:
    """トラッカーの生成（アソシエーション部品を差し替える）"""
    
    # 1. Gating (共通)
    gating_module = EllipsoidalGating(filter.H, filter.R, cfg.gate_threshold)

    # 2. Association (ここでアソシエーションアルゴリズムを切り替える)
    if cfg.tracker_type == "JPDA":
        association_logic = JPDAAssociation(
            filter, 
            clutter_density=cfg.clutter_density, 
            detection_prob=cfg.detection_prob
        )
    elif cfg.tracker_type == "GNN":
        # GNN用ロジックを使用
        association_logic = GNNAssociation(filter)
    else:
        raise ValueError(f"Unknown tracker type: {cfg.tracker_type}")

    # 3. トラッカー本体に注入 (クラスはStandardTrackerのままでOK)
    # StandardTrackerクラスは「アソシエーション結果(beta)を使って更新する」クラスなので、
    # betaが0/1(GNN)であっても数式上問題なく動作します。
    # 名前が気になる場合はクラス名を `StandardTracker` などに変えても良いですが、
    # そのままでも機能します。
    return StandardTracker(filter, gating_module, association_logic)