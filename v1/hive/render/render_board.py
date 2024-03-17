from jinja2 import Environment, PackageLoader, select_autoescape

env = Environment(
    loader=PackageLoader("hive.render"),
    autoescape=select_autoescape()
)

template = env.get_template("standalone_board.html")
rendered = template.render()
with open('test.html', "w") as fh:
    fh.write(rendered)