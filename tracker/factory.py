# factory.py
from config import Config
from core.kalman_filter import LinearKalmanFilter
from interfaces.tracker import ITracker

# 具体的な実装クラスをここでインポート
# 将来新しいアルゴリズムを追加するときは、ここにimportを追加するだけでOK
from modules.gatings.ellipsoidal import EllipsoidalGating
from modules.associations.jpda import JPDAAssociation
from modules.trackers.jpdaf import JPDAF

def create_tracker(cfg: Config, kf: LinearKalmanFilter) -> ITracker:
    """
    Configの設定値に基づいて、適切なTrackerインスタンスを生成する工場関数
    
    Args:
        cfg: 設定オブジェクト
        kf: 共通で使用するカルマンフィルタ
        
    Returns:
        完成したTracker (ITrackerインターフェースを持つもの)
    """
    
    # 1. ゲーティング部品の生成
    if cfg.gating_type == "Ellipsoidal":
        # 楕円ゲーティング
        gating_module = EllipsoidalGating(kf.H, kf.R, cfg.gate_threshold)
    
    elif cfg.gating_type == "Rectangular":
        # (将来の拡張用) 矩形ゲーティング
        # gating_module = RectangularGating(...)
        raise NotImplementedError("Rectangular gating is not implemented yet.")
    
    else:
        raise ValueError(f"Unknown gating type: {cfg.gating_type}")

    # 2. アソシエーション部品の生成 & トラッカーの組み立て
    if cfg.tracker_type == "JPDA":
        # JPDA用のアソシエーションロジックを作成
        association_logic = JPDAAssociation(
            kf, 
            clutter_density=cfg.clutter_density, 
            detection_prob=cfg.detection_prob
        )
        
        # 部品をJPDAFに注入して完成品を返す
        return JPDAF(kf, gating_module, association_logic)
    
    elif cfg.tracker_type == "GNN":
        # (将来の拡張用) GNNの実装例
        # association_logic = GNNAssociation(kf)
        # return GNNTracker(kf, gating_module, association_logic)
        raise NotImplementedError("GNN tracker is not implemented yet.")
        
    else:
        raise ValueError(f"Unknown tracker type: {cfg.tracker_type}")