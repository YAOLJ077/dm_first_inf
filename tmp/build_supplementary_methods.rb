#!/usr/bin/env ruby
# Convert the supplementary methods and apply journal-style Word formatting.
require 'fileutils'
require 'tmpdir'
require 'rexml/document'
require 'rexml/xpath'

ROOT = File.expand_path('..', __dir__)
SOURCE = File.join(ROOT, 'paper/submission/supplementary_methods.md')
OUTPUT = File.join(ROOT, 'paper/submission/supplementary.docx')
NS = {'w' => 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

def child(parent, name, attributes = {})
  element = parent.elements[name] || parent.add_element(name)
  if %w[w:p w:r w:tr w:tc].include?(parent.expanded_name) && name.end_with?('Pr')
    first = parent.elements[1]
    if first && first != element
      parent.delete_element(element)
      parent.insert_before(first, element)
    end
  end
  attributes.each { |key, value| element.add_attribute(key, value.to_s) }
  element
end

def save_xml(path, xml)
  File.open(path, 'w') { |file| xml.write(file) }
end

abort 'Pandoc conversion failed' unless system(
  'pandoc', SOURCE, '--from=markdown+tex_math_dollars', '--standalone',
  '--resource-path', ROOT,
  '-o', OUTPUT
)

Dir.mktmpdir('supplementary_docx_', '/private/tmp') do |directory|
  abort 'DOCX extraction failed' unless system('unzip', '-q', OUTPUT, '-d', directory)
  style_path = File.join(directory, 'word/styles.xml')
  styles = REXML::Document.new(File.read(style_path))
  REXML::XPath.each(styles, '//w:style', NS) do |style|
    id = style.attributes['w:styleId']
    properties = child(style, 'w:rPr')
    fonts = child(properties, 'w:rFonts', {
      'w:ascii' => 'Times New Roman', 'w:hAnsi' => 'Times New Roman',
      'w:eastAsia' => 'Times New Roman', 'w:cs' => 'Times New Roman'
    })
    %w[w:asciiTheme w:hAnsiTheme w:eastAsiaTheme w:cstheme].each { |attribute| fonts.attributes.delete(attribute) }
    colour = child(properties, 'w:color', 'w:val' => '000000')
    colour.attributes.delete('w:themeColor')
    colour.attributes.delete('w:themeShade')
    colour.attributes.delete('w:themeTint')
    size = {'Title' => 32, 'Heading1' => 26, 'Heading2' => 24}.fetch(id, 22)
    child(properties, 'w:sz', 'w:val' => size)
    child(properties, 'w:szCs', 'w:val' => size)
    paragraph = child(style, 'w:pPr')
    child(paragraph, 'w:spacing', 'w:after' => 120, 'w:line' => 264, 'w:lineRule' => 'auto')
    if id == 'Compact'
      child(paragraph, 'w:spacing', 'w:before' => 0, 'w:after' => 60, 'w:line' => 240)
    end
    if id == 'Title' || id.start_with?('Heading')
      child(paragraph, 'w:keepNext')
      child(paragraph, 'w:spacing', 'w:before' => 180, 'w:after' => 100)
      paragraph.delete_element('w:pBdr')
      properties.delete_element('w:u')
      child(properties, 'w:b') if id.start_with?('Heading')
    end
  end
  save_xml(style_path, styles)

  document_path = File.join(directory, 'word/document.xml')
  document = REXML::Document.new(File.read(document_path))
  REXML::XPath.each(document, '//w:sectPr', NS) do |section|
    child(section, 'w:pgSz', 'w:w' => 11906, 'w:h' => 16838)
    child(section, 'w:pgMar', {
      'w:top' => 1276, 'w:bottom' => 1276, 'w:left' => 1276,
      'w:right' => 1276, 'w:header' => 600, 'w:footer' => 600
    })
  end
  table_widths = [[4400, 1500, 1800, 1654], [2100, 4154, 1550, 1550],
                  [2254, 900, 2500, 1850, 1850], [1700, 5654, 2000]]
  REXML::XPath.match(document, '//w:tbl', NS).each_with_index do |table, table_index|
    properties = child(table, 'w:tblPr')
    child(properties, 'w:tblW', 'w:w' => 9354, 'w:type' => 'dxa')
    widths = table_widths.fetch(table_index)
    table.elements['w:tblGrid'].elements.to_a.each_with_index do |column, index|
      column.add_attribute('w:w', widths.fetch(index).to_s)
    end
    borders = child(properties, 'w:tblBorders')
    %w[top left bottom right insideH insideV].each do |side|
      child(borders, "w:#{side}", 'w:val' => 'single', 'w:sz' => 4, 'w:color' => 'D9D9D9')
    end
    table.elements['w:tr'].elements.each('w:tc') do |cell|
      child(child(cell, 'w:tcPr'), 'w:shd', 'w:fill' => 'E7E7E7')
    end
    table.elements.each('w:tr') do |row|
      child(child(row, 'w:trPr'), 'w:cantSplit')
      row.elements.to_a.select { |element| element.expanded_name == 'w:tc' }.each_with_index do |cell, index|
        child(child(cell, 'w:tcPr'), 'w:tcW', 'w:w' => widths.fetch(index), 'w:type' => 'dxa')
      end
    end
    if table_index == 3
      rows = table.elements.to_a.select { |element| element.expanded_name == 'w:tr' }
      rows[0...-1].each do |row|
        REXML::XPath.each(row, './/w:p', NS) do |paragraph|
          child(child(paragraph, 'w:pPr'), 'w:keepNext')
        end
      end
    end
  end
  REXML::XPath.each(document, '//w:body/w:p', NS) do |paragraph|
    text = REXML::XPath.match(paragraph, './/w:t', NS).map(&:text).join
    if text.start_with?('Table S')
      child(child(paragraph, 'w:pPr'), 'w:keepNext')
    end
    if REXML::XPath.first(paragraph, './/w:drawing', NS)
      properties = child(paragraph, 'w:pPr')
      child(properties, 'w:keepNext')
      child(properties, 'w:jc', 'w:val' => 'center')
    end
  end
  save_xml(document_path, document)
  Dir.chdir(directory) do
    abort 'DOCX packaging failed' unless system('zip', '-q', '-u', OUTPUT, 'word/styles.xml', 'word/document.xml')
  end
end

puts "Created #{OUTPUT}"
