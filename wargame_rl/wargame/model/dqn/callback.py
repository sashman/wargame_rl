from pytorch_lightning.callbacks import ModelCheckpoint


def get_checkpoint_callback(name: str):
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./checkpoints/{name}",
        filename="dqn-{epoch:02d}-{total_reward:.3f}",
        save_top_k=3,
        monitor="total_reward",
        mode="max",
        save_last=True,
    )
    return [checkpoint_callback]
