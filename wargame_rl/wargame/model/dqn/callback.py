from pytorch_lightning.callbacks import ModelCheckpoint


def get_checkpoint_callback():
    # Create checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath="./checkpoints",
        filename="dqn-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        monitor="total_reward",
        mode="max",
        save_last=True,
    )
    return [checkpoint_callback]
