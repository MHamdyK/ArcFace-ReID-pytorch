import click
from src.train import main as train_cmd
from src.infer import main as infer_group

@click.group()
def cli():
    """ArcFace‑ReID command‑line interface."""
    pass

cli.add_command(train_cmd, name='train')
cli.add_command(infer_group, name='infer')

if __name__ == '__main__':
    cli()
