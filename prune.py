from ultralytics import YOLO

# base model path
MODEL_PATH = "YOLO11s_best.pt"

def prunetrain(
    model,
    train_epochs,
    prune_epochs=0,
    quick_pruning=True,
    prune_ratio=0.5,
    prune_iterative_steps=1,
    data='Wayang-Kulit-Detc-13/data.yaml',
    name='yolo11',
    imgsz=640,
    batch=8,
    device=[0],
    sparse_training=False,
    lr0=0.001,
    optimizer="SGD"
):
    if not quick_pruning:
        if train_epochs > 0:
            model.train(
                data=data,
                epochs=train_epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                name=name,
                prune=False,
                sparse_training=sparse_training,
                lr0=lr0,
                optimizer=optimizer
            )

        if prune_epochs > 0:
            return model.train(
                data=data,
                epochs=prune_epochs,
                imgsz=imgsz,
                batch=batch,
                device=device,
                name=name,
                prune=True,
                prune_ratio=prune_ratio,
                prune_iterative_steps=prune_iterative_steps,
                lr0=lr0,
                optimizer=optimizer
            )
    else:
        return model.train(
            data=data,
            epochs=train_epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            name=name,
            prune=True,
            prune_ratio=prune_ratio,
            prune_iterative_steps=prune_iterative_steps,
            lr0=lr0,
            optimizer=optimizer
        )


prune_ratios = [0.3, 0.4, 0.5]

for ratio in prune_ratios:
    print(f"\n🚀 Running pruning with ratio = {ratio}\n")
    model = YOLO(MODEL_PATH)

    prunetrain(
        model=model,
        quick_pruning=True,
        data='Wayang-Kulit-Detc-13/data.yaml',
        train_epochs=100,
        imgsz=640,
        batch=8,
        device=[0],
        name=f'yolo11s_prune_{int(ratio*100)}',  
        lr0=0.0005,
        optimizer="SGD",
        prune_ratio=ratio,
        prune_iterative_steps=1,
        sparse_training=False
    )