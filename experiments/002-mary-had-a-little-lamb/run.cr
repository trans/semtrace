require "../../src/semtrace"

MARY = <<-TEXT
Mary had a little lamb, its fleece was white as snow, and everywhere that Mary went, the lamb was sure to go. It followed her to school one day, which was against the rules. It made the children laugh and play to see a lamb at school.
TEXT

args = Semtrace::RealText.parse_args({
  "mary" => {MARY, "Mary Had a Little Lamb"},
})

text = args[:text].empty? ? MARY : args[:text]
label = args[:label].empty? ? "Mary Had a Little Lamb" : args[:label]
data_dir = args[:data_dir].empty? ? (Path[__DIR__].parent.parent / "data").to_s : args[:data_dir]

Semtrace::RealText.run(text, label, data_dir, trace: args[:trace])
