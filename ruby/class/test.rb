#!/usr/bin/ruby

class Person
  VAR = "const_person"
  @@var = "class_person"
  def initialize(name)
    @name = name
  end

  def getName()
    return @name
  end

  def setName(name)
    @name = name
  end
  
  def getVar()
    return @@var
  end

  def getVAR()
    return VAR
  end

  attr_reader :name
end

class SubPerson < Person
  def showStaticVar()
    puts(@@var)
  end
end

puts(Person::VAR)
hoge = Person.new("Sato")
print "get @name from accessor", "\n";
puts(hoge.name)

print "get @name from accessor method\n";
puts(hoge.getName())

puts(hoge.getVar())
hoge.setName("Kaminishi")
puts(hoge.getName())

foo = SubPerson.new("Ishi")
foo.showStaticVar()
puts(foo.getName());

puts()

puts(hoge.getVAR())
