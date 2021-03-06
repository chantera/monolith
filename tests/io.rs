extern crate monolith;
#[macro_use]
extern crate serde_derive;
extern crate tempfile;

#[cfg(feature = "app")]
use monolith::io::cache::Cache;
#[cfg(feature = "serialize")]
use monolith::io::serialize;

#[cfg(feature = "serialize")]
#[test]
fn test_serialize() {
    #[derive(Debug, PartialEq, Deserialize, Serialize)]
    struct Person {
        name: String,
        age: u32,
    }

    impl Person {
        fn new<S: Into<String>>(name: S, age: u32) -> Self {
            Person {
                name: name.into(),
                age: age,
            }
        }
    }

    let person1 = Person::new("John", 26);
    let person2 = Person::new("Mary", 23);
    let person3 = Person::new("Bob", 30);

    {
        let bytes = serialize::serialize(&person1, serialize::Format::Json).unwrap();
        let person1_de: Person = serialize::deserialize(&bytes, serialize::Format::Json).unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "{\"name\":\"John\",\"age\":26}"
        );
        assert_eq!(person1, person1_de);
    }
    {
        let bytes = serialize::serialize(&person1, serialize::Format::JsonPretty).unwrap();
        let person1_de: Person =
            serialize::deserialize(&bytes, serialize::Format::JsonPretty).unwrap();
        assert_eq!(
            String::from_utf8(bytes).unwrap(),
            "{\n  \"name\": \"John\",\n  \"age\": 26\n}"
        );
        assert_eq!(person1, person1_de);
    }
    {
        let bytes = serialize::serialize(&person1, serialize::Format::Msgpack).unwrap();
        let person1_de: Person =
            serialize::deserialize(&bytes, serialize::Format::Msgpack).unwrap();
        assert_eq!(bytes, [146, 164, 74, 111, 104, 110, 26]);
        assert_eq!(person1, person1_de);
    }

    let people = vec![person1, person2, person3];

    {
        let file = tempfile::NamedTempFile::new().unwrap();
        serialize::write_to(&people, file.path(), serialize::Format::Json).unwrap();
        let objs: Vec<Person> = serialize::read_from(file.path(), serialize::Format::Json).unwrap();
        assert_eq!(objs.len(), people.len());
        for (obj, person) in objs.iter().zip(&people) {
            assert_eq!(obj, person);
        }
        file.close().unwrap();
    }
}

#[cfg(feature = "app")]
#[test]
fn test_cache() {
    #[derive(Debug, PartialEq, Deserialize, Serialize)]
    struct Person {
        name: String,
        age: u32,
    }

    impl Person {
        fn new<S: Into<String>>(name: S, age: u32) -> Self {
            Person {
                name: name.into(),
                age: age,
            }
        }
    }

    let person1 = Person::new("John", 26);

    let mut cache = Cache::new("test_cache");
    cache.write("John", &person1).unwrap();
    assert!(cache.exists("John"));
    let obj: Person = cache.read("John").unwrap();
    assert_eq!(obj, person1);
}

#[cfg(feature = "dataset-conll")]
mod tests {
    use std::io::{Cursor, SeekFrom};

    use monolith::dataset::conll::{ConllTokenized, Reader};
    use monolith::io::prelude::*;
    use monolith::lang::prelude::*;

    #[test]
    fn test_parse_conll() {
        let mut reader = Reader::new(Cursor::new(RAW_TEXT));
        let mut buf = vec![];

        assert_eq!(reader.read_upto(2, &mut buf).unwrap(), 2);
        println!("buf[0]: {}", buf[0]);
        assert_eq!(buf[0][0].form(), "<ROOT>");
        assert_eq!(buf[0][3].cpostag(), Some("IN"));
        assert_eq!(buf[0][6].form(), "Ways");
        assert_eq!(buf[0][11].head(), Some(10));
        assert_eq!(buf[0][11].phead(), None);
        assert_eq!(buf[0][25].postag(), Some("VBG"));

        println!("buf[1]: {}", buf[1]);
        assert_eq!(buf[1][10].form(), "-LRB-");
        assert_eq!(buf[1][12].deprel(), Some("punct"));
        assert_eq!(buf[1][12].pdeprel(), None);
        assert_eq!(buf[1][41].feats(), None);

        assert_eq!(reader.read_upto(2, &mut buf).unwrap(), 2);
        assert_eq!(reader.read_upto(2, &mut buf).unwrap(), 0);
        assert_eq!(buf.len(), 4);

        buf.clear();
        assert_eq!(reader.seek(SeekFrom::Start(0)).unwrap(), 0);
        assert_eq!(reader.read_to_end(&mut buf).unwrap(), 4);
        assert_eq!(reader.read_upto(1, &mut buf).unwrap(), 0);
        assert_eq!(buf.len(), 4);
    }

    static RAW_TEXT: &'static str = r#"
1	Influential	_	JJ	JJ	_	2	amod	_	_
2	members	_	NNS	NNS	_	10	nsubj	_	_
3	of	_	IN	IN	_	2	prep	_	_
4	the	_	DT	DT	_	6	det	_	_
5	House	_	NNP	NNP	_	6	nn	_	_
6	Ways	_	NNP	NNP	_	3	pobj	_	_
7	and	_	CC	CC	_	6	cc	_	_
8	Means	_	NNP	NNP	_	9	nn	_	_
9	Committee	_	NNP	NNP	_	6	conj	_	_
10	introduced	_	VBD	VBD	_	0	root	_	_
11	legislation	_	NN	NN	_	10	dobj	_	_
12	that	_	WDT	WDT	_	14	nsubj	_	_
13	would	_	MD	MD	_	14	aux	_	_
14	restrict	_	VB	VB	_	11	rcmod	_	_
15	how	_	WRB	WRB	_	22	advmod	_	_
16	the	_	DT	DT	_	20	det	_	_
17	new	_	JJ	JJ	_	20	amod	_	_
18	savings-and-loan	_	NN	NN	_	20	nn	_	_
19	bailout	_	NN	NN	_	20	nn	_	_
20	agency	_	NN	NN	_	22	nsubj	_	_
21	can	_	MD	MD	_	22	aux	_	_
22	raise	_	VB	VB	_	14	ccomp	_	_
23	capital	_	NN	NN	_	22	dobj	_	_
24	,	_	,	,	_	14	punct	_	_
25	creating	_	VBG	VBG	_	14	xcomp	_	_
26	another	_	DT	DT	_	28	det	_	_
27	potential	_	JJ	JJ	_	28	amod	_	_
28	obstacle	_	NN	NN	_	25	dobj	_	_
29	to	_	TO	TO	_	28	prep	_	_
30	the	_	DT	DT	_	31	det	_	_
31	government	_	NN	NN	_	33	poss	_	_
32	's	_	POS	POS	_	31	possessive	_	_
33	sale	_	NN	NN	_	29	pobj	_	_
34	of	_	IN	IN	_	33	prep	_	_
35	sick	_	JJ	JJ	_	36	amod	_	_
36	thrifts	_	NNS	NNS	_	34	pobj	_	_
37	.	_	.	.	_	10	punct	_	_

1	The	_	DT	DT	_	2	det	_	_
2	bill	_	NN	NN	_	17	nsubj	_	_
3	,	_	,	,	_	2	punct	_	_
4	whose	_	WP$	WP$	_	5	poss	_	_
5	backers	_	NNS	NNS	_	6	nsubj	_	_
6	include	_	VBP	VBP	_	2	rcmod	_	_
7	Chairman	_	NNP	NNP	_	9	nn	_	_
8	Dan	_	NNP	NNP	_	9	nn	_	_
9	Rostenkowski	_	NNP	NNP	_	6	dobj	_	_
10	-LRB-	_	-LRB-	-LRB-	_	11	punct	_	_
11	D.	_	NNP	NNP	_	9	appos	_	_
12	,	_	,	,	_	11	punct	_	_
13	Ill.	_	NNP	NNP	_	11	dep	_	_
14	-RRB-	_	-RRB-	-RRB-	_	11	punct	_	_
15	,	_	,	,	_	2	punct	_	_
16	would	_	MD	MD	_	17	aux	_	_
17	prevent	_	VB	VB	_	0	root	_	_
18	the	_	DT	DT	_	21	det	_	_
19	Resolution	_	NNP	NNP	_	21	nn	_	_
20	Trust	_	NNP	NNP	_	21	nn	_	_
21	Corp.	_	NNP	NNP	_	17	dobj	_	_
22	from	_	IN	IN	_	17	prep	_	_
23	raising	_	VBG	VBG	_	22	pcomp	_	_
24	temporary	_	JJ	JJ	_	26	amod	_	_
25	working	_	VBG	VBG	_	26	amod	_	_
26	capital	_	NN	NN	_	23	dobj	_	_
27	by	_	IN	IN	_	17	prep	_	_
28	having	_	VBG	VBG	_	27	pcomp	_	_
29	an	_	DT	DT	_	31	det	_	_
30	RTC-owned	_	JJ	JJ	_	31	amod	_	_
31	bank	_	NN	NN	_	28	dobj	_	_
32	or	_	CC	CC	_	31	cc	_	_
33	thrift	_	NN	NN	_	35	nn	_	_
34	issue	_	NN	NN	_	35	nn	_	_
35	debt	_	NN	NN	_	31	conj	_	_
36	that	_	WDT	WDT	_	40	nsubjpass	_	_
37	would	_	MD	MD	_	40	aux	_	_
38	n't	_	RB	RB	_	40	neg	_	_
39	be	_	VB	VB	_	40	auxpass	_	_
40	counted	_	VBN	VBN	_	31	rcmod	_	_
41	on	_	IN	IN	_	40	prep	_	_
42	the	_	DT	DT	_	44	det	_	_
43	federal	_	JJ	JJ	_	44	amod	_	_
44	budget	_	NN	NN	_	41	pobj	_	_
45	.	_	.	.	_	17	punct	_	_

1	The	_	DT	DT	_	2	det	_	_
2	bill	_	NN	NN	_	3	nsubj	_	_
3	intends	_	VBZ	VBZ	_	0	root	_	_
4	to	_	TO	TO	_	5	aux	_	_
5	restrict	_	VB	VB	_	3	xcomp	_	_
6	the	_	DT	DT	_	7	det	_	_
7	RTC	_	NNP	NNP	_	5	dobj	_	_
8	to	_	TO	TO	_	5	prep	_	_
9	Treasury	_	NNP	NNP	_	10	nn	_	_
10	borrowings	_	NNS	NNS	_	8	pobj	_	_
11	only	_	RB	RB	_	10	advmod	_	_
12	,	_	,	,	_	3	punct	_	_
13	unless	_	IN	IN	_	16	mark	_	_
14	the	_	DT	DT	_	15	det	_	_
15	agency	_	NN	NN	_	16	nsubj	_	_
16	receives	_	VBZ	VBZ	_	3	advcl	_	_
17	specific	_	JJ	JJ	_	19	amod	_	_
18	congressional	_	JJ	JJ	_	19	amod	_	_
19	authorization	_	NN	NN	_	16	dobj	_	_
20	.	_	.	.	_	3	punct	_	_

1	``	_	``	``	_	22	punct	_	_
2	Such	_	JJ	JJ	_	7	amod	_	_
3	agency	_	NN	NN	_	7	nn	_	_
4	`	_	``	``	_	7	punct	_	_
5	self-help	_	NN	NN	_	7	nn	_	_
6	'	_	''	''	_	7	punct	_	_
7	borrowing	_	NN	NN	_	8	nsubj	_	_
8	is	_	VBZ	VBZ	_	22	ccomp	_	_
9	unauthorized	_	JJ	JJ	_	8	acomp	_	_
10	and	_	CC	CC	_	9	cc	_	_
11	expensive	_	JJ	JJ	_	9	conj	_	_
12	,	_	,	,	_	9	punct	_	_
13	far	_	RB	RB	_	15	advmod	_	_
14	more	_	RBR	RBR	_	15	advmod	_	_
15	expensive	_	JJ	JJ	_	9	dep	_	_
16	than	_	IN	IN	_	15	prep	_	_
17	direct	_	JJ	JJ	_	19	amod	_	_
18	Treasury	_	NNP	NNP	_	19	nn	_	_
19	borrowing	_	NN	NN	_	16	pobj	_	_
20	,	_	,	,	_	22	punct	_	_
21	''	_	''	''	_	22	punct	_	_
22	said	_	VBD	VBD	_	0	root	_	_
23	Rep.	_	NNP	NNP	_	25	nn	_	_
24	Fortney	_	NNP	NNP	_	25	nn	_	_
25	Stark	_	NNP	NNP	_	22	nsubj	_	_
26	-LRB-	_	-LRB-	-LRB-	_	27	punct	_	_
27	D.	_	NNP	NNP	_	25	appos	_	_
28	,	_	,	,	_	27	punct	_	_
29	Calif.	_	NNP	NNP	_	27	dep	_	_
30	-RRB-	_	-RRB-	-RRB-	_	27	punct	_	_
31	,	_	,	,	_	25	punct	_	_
32	the	_	DT	DT	_	33	det	_	_
33	bill	_	NN	NN	_	36	poss	_	_
34	's	_	POS	POS	_	33	possessive	_	_
35	chief	_	JJ	JJ	_	36	amod	_	_
36	sponsor	_	NN	NN	_	25	appos	_	_
37	.	_	.	.	_	22	punct	_	_

"#;

}
