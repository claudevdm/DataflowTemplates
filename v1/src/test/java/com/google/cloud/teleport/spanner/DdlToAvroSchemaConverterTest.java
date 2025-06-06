/*
 * Copyright (C) 2018 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not
 * use this file except in compliance with the License. You may obtain a copy of
 * the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations under
 * the License.
 */
package com.google.cloud.teleport.spanner;

import static com.google.cloud.teleport.spanner.AvroUtil.DEFAULT_EXPRESSION;
import static com.google.cloud.teleport.spanner.AvroUtil.GENERATION_EXPRESSION;
import static com.google.cloud.teleport.spanner.AvroUtil.GOOGLE_FORMAT_VERSION;
import static com.google.cloud.teleport.spanner.AvroUtil.GOOGLE_STORAGE;
import static com.google.cloud.teleport.spanner.AvroUtil.HIDDEN;
import static com.google.cloud.teleport.spanner.AvroUtil.IDENTITY_COLUMN;
import static com.google.cloud.teleport.spanner.AvroUtil.INPUT;
import static com.google.cloud.teleport.spanner.AvroUtil.NOT_NULL;
import static com.google.cloud.teleport.spanner.AvroUtil.OUTPUT;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_CHANGE_STREAM_FOR_CLAUSE;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_CHECK_CONSTRAINT;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_EDGE_TABLE;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_ENTITY;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_ENTITY_MODEL;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_ENTITY_PLACEMENT;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_ENTITY_PROPERTY_GRAPH;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_FOREIGN_KEY;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_INDEX;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_INTERLEAVE_TYPE;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_LABEL;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_NODE_TABLE;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_ON_DELETE_ACTION;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_OPTION;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_PARENT;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_PLACEMENT_KEY;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_PRIMARY_KEY;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_PROPERTY_DECLARATION;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_REMOTE;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_SEQUENCE_COUNTER_START;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_SEQUENCE_KIND;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_SEQUENCE_SKIP_RANGE_MAX;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_SEQUENCE_SKIP_RANGE_MIN;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_UDF_DEFINITION;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_UDF_NAME;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_UDF_PARAMETER;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_UDF_SECURITY;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_UDF_TYPE;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_VIEW_QUERY;
import static com.google.cloud.teleport.spanner.AvroUtil.SPANNER_VIEW_SECURITY;
import static com.google.cloud.teleport.spanner.AvroUtil.SQL_TYPE;
import static com.google.cloud.teleport.spanner.AvroUtil.STORED;
import static org.hamcrest.Matchers.empty;
import static org.hamcrest.Matchers.equalTo;
import static org.hamcrest.Matchers.hasSize;
import static org.hamcrest.Matchers.notNullValue;
import static org.hamcrest.Matchers.nullValue;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertThat;

import com.google.cloud.spanner.Dialect;
import com.google.cloud.teleport.spanner.common.NumericUtils;
import com.google.cloud.teleport.spanner.common.Type;
import com.google.cloud.teleport.spanner.common.Type.StructField;
import com.google.cloud.teleport.spanner.ddl.Ddl;
import com.google.cloud.teleport.spanner.ddl.GraphElementTable;
import com.google.cloud.teleport.spanner.ddl.GraphElementTable.GraphNodeTableReference;
import com.google.cloud.teleport.spanner.ddl.GraphElementTable.LabelToPropertyDefinitions;
import com.google.cloud.teleport.spanner.ddl.GraphElementTable.PropertyDefinition;
import com.google.cloud.teleport.spanner.ddl.PropertyGraph;
import com.google.cloud.teleport.spanner.ddl.Table.InterleaveType;
import com.google.cloud.teleport.spanner.ddl.Udf.SqlSecurity;
import com.google.cloud.teleport.spanner.ddl.UdfParameter;
import com.google.cloud.teleport.spanner.ddl.View;
import com.google.common.collect.ImmutableList;
import java.util.Arrays;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import org.apache.avro.LogicalTypes;
import org.apache.avro.Schema;
import org.apache.avro.SchemaBuilder;
import org.junit.Test;

/** Test for {@link DdlToAvroSchemaConverter}. */
public class DdlToAvroSchemaConverterTest {

  @Test
  public void emptyDb() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl empty = Ddl.builder().build();
    assertThat(converter.convert(empty), empty());
  }

  @Test
  public void pgEmptyDb() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl empty = Ddl.builder(Dialect.POSTGRESQL).build();
    assertThat(converter.convert(empty), empty());
  }

  @Test
  public void simple() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder()
            .createTable("Users")
            .column("id")
            .int64()
            .notNull()
            .endColumn()
            .column("first_name")
            .string()
            .size(10)
            .defaultExpression("'John'")
            .endColumn()
            .column("last_name")
            .type(Type.string())
            .max()
            .endColumn()
            .column("full_name")
            .type(Type.string())
            .max()
            .generatedAs("CONCAT(first_name, ' ', last_name)")
            .stored()
            .endColumn()
            .column("gen_id")
            .int64()
            .notNull()
            .generatedAs("MOD(id+1, 64)")
            .stored()
            .endColumn()
            .column("MyTokens")
            .tokenlist()
            .isHidden(true)
            .generatedAs("(TOKENIZE_FULLTEXT(MyData))")
            .endColumn()
            .column("Embeddings")
            .type(Type.array(Type.float32()))
            .arrayLength(Integer.valueOf(128))
            .endColumn()
            .column("HiddenColumn")
            .type(Type.string())
            .max()
            .isHidden(true)
            .endColumn()
            .column("identity_column")
            .type(Type.int64())
            .isIdentityColumn(true)
            .sequenceKind("bit_reversed_positive")
            .counterStartValue(1000L)
            .skipRangeMin(2000L)
            .skipRangeMax(3000L)
            .endColumn()
            .column("identity_column_no_kind")
            .type(Type.int64())
            .isIdentityColumn(true)
            .counterStartValue(1000L)
            .skipRangeMin(2000L)
            .skipRangeMax(3000L)
            .endColumn()
            .column("uuid_column")
            .type(Type.uuid())
            .endColumn()
            .column("identity_column_no_params")
            .type(Type.int64())
            .isIdentityColumn(true)
            .endColumn()
            .primaryKey()
            .asc("id")
            .asc("gen_id")
            .desc("last_name")
            .end()
            .indexes(
                ImmutableList.of(
                    "CREATE INDEX `UsersByFirstName` ON `Users` (`first_name`)",
                    "CREATE SEARCH INDEX `SearchIndex` ON `Users` (`MyTokens`)"
                        + " OPTIONS (sort_order_sharding=TRUE)",
                    "CREATE VECTOR INDEX `VI` ON `Users`(`Embeddings` ) WHERE Embeddings IS NOT NULL"
                        + " OPTIONS (distance_type=\"COSINE\")"))
            .foreignKeys(
                ImmutableList.of(
                    "ALTER TABLE `Users` ADD CONSTRAINT `fk` FOREIGN KEY (`first_name`)"
                        + " REFERENCES `AllowedNames` (`first_name`)",
                    "ALTER TABLE `Users` ADD CONSTRAINT `fk_odc` FOREIGN KEY (`last_name`)"
                        + " REFERENCES `AllowedNames` (`last_name`) ON DELETE CASCADE",
                    "ALTER TABLE `Users` ADD CONSTRAINT `fk_not_enforced_no_action`"
                        + " FOREIGN KEY (`last_name`) REFERENCES "
                        + "`AllowedNames` (`last_name`) ON DELETE NO ACTION NOT ENFORCED",
                    "ALTER TABLE `Users` ADD CONSTRAINT `fk_enforced`"
                        + " FOREIGN KEY (`last_name`) REFERENCES "
                        + "`AllowedNames` (`last_name`) ENFORCED"))
            .checkConstraints(ImmutableList.of("CONSTRAINT ck CHECK (`first_name` != `last_name`)"))
            .endTable()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroSchema = result.iterator().next();

    assertThat(avroSchema.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));

    assertThat(avroSchema.getName(), equalTo("Users"));

    List<Schema.Field> fields = avroSchema.getFields();

    assertThat(fields, hasSize(12));

    assertThat(fields.get(0).name(), equalTo("id"));
    // Not null
    assertThat(fields.get(0).schema().getType(), equalTo(Schema.Type.LONG));
    assertThat(fields.get(0).getProp(SQL_TYPE), equalTo("INT64"));
    assertThat(fields.get(0).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(0).getProp(GENERATION_EXPRESSION), equalTo(null));
    assertThat(fields.get(0).getProp(STORED), equalTo(null));
    assertThat(fields.get(0).getProp(DEFAULT_EXPRESSION), equalTo(null));

    assertThat(fields.get(1).name(), equalTo("first_name"));
    assertThat(fields.get(1).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(1).getProp(SQL_TYPE), equalTo("STRING(10)"));
    assertThat(fields.get(1).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(1).getProp(GENERATION_EXPRESSION), equalTo(null));
    assertThat(fields.get(1).getProp(STORED), equalTo(null));
    assertThat(fields.get(1).getProp(DEFAULT_EXPRESSION), equalTo("'John'"));

    assertThat(fields.get(2).name(), equalTo("last_name"));
    assertThat(fields.get(2).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(2).getProp(SQL_TYPE), equalTo("STRING(MAX)"));
    assertThat(fields.get(2).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(2).getProp(GENERATION_EXPRESSION), equalTo(null));
    assertThat(fields.get(2).getProp(STORED), equalTo(null));
    assertThat(fields.get(2).getProp(DEFAULT_EXPRESSION), equalTo(null));

    assertThat(fields.get(3).name(), equalTo("full_name"));
    assertThat(fields.get(3).schema(), equalTo(Schema.create(Schema.Type.NULL)));
    assertThat(fields.get(3).getProp(SQL_TYPE), equalTo("STRING(MAX)"));
    assertThat(fields.get(3).getProp(NOT_NULL), equalTo("false"));
    assertThat(
        fields.get(3).getProp(GENERATION_EXPRESSION),
        equalTo("CONCAT(first_name, ' ', last_name)"));
    assertThat(fields.get(3).getProp(STORED), equalTo("true"));
    assertThat(fields.get(3).getProp(DEFAULT_EXPRESSION), equalTo(null));

    assertThat(fields.get(4).name(), equalTo("gen_id"));
    assertThat(fields.get(4).schema(), equalTo(Schema.create(Schema.Type.NULL)));
    assertThat(fields.get(4).getProp(SQL_TYPE), equalTo("INT64"));
    assertThat(fields.get(4).getProp(NOT_NULL), equalTo("true"));
    assertThat(fields.get(4).getProp(GENERATION_EXPRESSION), equalTo("MOD(id+1, 64)"));
    assertThat(fields.get(4).getProp(STORED), equalTo("true"));
    assertThat(fields.get(4).getProp(DEFAULT_EXPRESSION), equalTo(null));

    assertThat(fields.get(5).name(), equalTo("MyTokens"));
    assertThat(fields.get(5).schema(), equalTo(Schema.create(Schema.Type.NULL)));
    assertThat(fields.get(5).getProp(SQL_TYPE), equalTo("TOKENLIST"));
    assertThat(fields.get(5).getProp(NOT_NULL), equalTo("false"));
    assertThat(fields.get(5).getProp(STORED), equalTo("false"));
    assertThat(fields.get(5).getProp(HIDDEN), equalTo("true"));
    assertThat(
        fields.get(5).getProp(GENERATION_EXPRESSION), equalTo("(TOKENIZE_FULLTEXT(MyData))"));
    assertThat(fields.get(5).getProp(DEFAULT_EXPRESSION), equalTo(null));

    assertThat(fields.get(6).name(), equalTo("Embeddings"));
    assertThat(fields.get(6).schema(), equalTo(nullableArray(Schema.Type.FLOAT)));
    assertThat(fields.get(6).getProp(SQL_TYPE), equalTo("ARRAY<FLOAT32>(vector_length=>128)"));
    assertThat(fields.get(6).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(6).getProp(STORED), equalTo(null));

    assertThat(fields.get(7).name(), equalTo("HiddenColumn"));
    assertThat(fields.get(7).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(7).getProp(SQL_TYPE), equalTo("STRING(MAX)"));
    assertThat(fields.get(7).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(7).getProp(GENERATION_EXPRESSION), equalTo(null));
    assertThat(fields.get(7).getProp(STORED), equalTo(null));
    assertThat(fields.get(7).getProp(HIDDEN), equalTo("true"));

    assertThat(fields.get(8).name(), equalTo("identity_column"));
    assertThat(fields.get(8).schema(), equalTo(nullableUnion(Schema.Type.LONG)));
    assertThat(fields.get(8).getProp(SQL_TYPE), equalTo("INT64"));
    assertThat(fields.get(8).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(8).getProp(IDENTITY_COLUMN), equalTo("true"));
    assertThat(fields.get(8).getProp(SPANNER_SEQUENCE_KIND), equalTo("bit_reversed_positive"));
    assertThat(fields.get(8).getProp(SPANNER_SEQUENCE_COUNTER_START), equalTo("1000"));
    assertThat(fields.get(8).getProp(SPANNER_SEQUENCE_SKIP_RANGE_MIN), equalTo("2000"));
    assertThat(fields.get(8).getProp(SPANNER_SEQUENCE_SKIP_RANGE_MAX), equalTo("3000"));

    assertThat(fields.get(9).name(), equalTo("identity_column_no_kind"));
    assertThat(fields.get(9).schema(), equalTo(nullableUnion(Schema.Type.LONG)));
    assertThat(fields.get(9).getProp(SQL_TYPE), equalTo("INT64"));
    assertThat(fields.get(9).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(9).getProp(IDENTITY_COLUMN), equalTo("true"));
    assertThat(fields.get(9).getProp(SPANNER_SEQUENCE_KIND), equalTo(null));
    assertThat(fields.get(9).getProp(SPANNER_SEQUENCE_COUNTER_START), equalTo("1000"));
    assertThat(fields.get(9).getProp(SPANNER_SEQUENCE_SKIP_RANGE_MIN), equalTo("2000"));
    assertThat(fields.get(9).getProp(SPANNER_SEQUENCE_SKIP_RANGE_MAX), equalTo("3000"));

    Schema.Field field10 = fields.get(10);
    assertThat(field10.name(), equalTo("uuid_column"));
    assertThat(field10.schema(), equalTo(nullableUuid()));
    assertThat(field10.getProp(SQL_TYPE), equalTo("UUID"));
    assertThat(field10.getProp(NOT_NULL), equalTo(null));
    assertThat(field10.getProp(GENERATION_EXPRESSION), equalTo(null));
    assertThat(field10.getProp(STORED), equalTo(null));
    assertThat(field10.getProp(DEFAULT_EXPRESSION), equalTo(null));

    assertThat(fields.get(11).name(), equalTo("identity_column_no_params"));
    assertThat(fields.get(11).schema(), equalTo(nullableUnion(Schema.Type.LONG)));
    assertThat(fields.get(11).getProp(SQL_TYPE), equalTo("INT64"));
    assertThat(fields.get(11).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(11).getProp(IDENTITY_COLUMN), equalTo("true"));
    assertThat(fields.get(11).getProp(SPANNER_SEQUENCE_KIND), equalTo(null));
    assertThat(fields.get(11).getProp(SPANNER_SEQUENCE_COUNTER_START), equalTo(null));
    assertThat(fields.get(11).getProp(SPANNER_SEQUENCE_SKIP_RANGE_MIN), equalTo(null));
    assertThat(fields.get(11).getProp(SPANNER_SEQUENCE_SKIP_RANGE_MAX), equalTo(null));

    // spanner pk
    assertThat(avroSchema.getProp(SPANNER_PRIMARY_KEY + "_0"), equalTo("`id` ASC"));
    assertThat(avroSchema.getProp(SPANNER_PRIMARY_KEY + "_1"), equalTo("`gen_id` ASC"));
    assertThat(avroSchema.getProp(SPANNER_PRIMARY_KEY + "_2"), equalTo("`last_name` DESC"));
    assertThat(avroSchema.getProp(SPANNER_PARENT), nullValue());
    assertThat(avroSchema.getProp(SPANNER_ON_DELETE_ACTION), nullValue());

    assertThat(
        avroSchema.getProp(SPANNER_INDEX + "0"),
        equalTo("CREATE INDEX `UsersByFirstName` ON `Users` (`first_name`)"));
    assertThat(
        avroSchema.getProp(SPANNER_INDEX + "1"),
        equalTo(
            "CREATE SEARCH INDEX `SearchIndex` ON `Users` (`MyTokens`)"
                + " OPTIONS (sort_order_sharding=TRUE)"));
    assertThat(
        avroSchema.getProp(SPANNER_FOREIGN_KEY + "0"),
        equalTo(
            "ALTER TABLE `Users` ADD CONSTRAINT `fk` FOREIGN KEY (`first_name`)"
                + " REFERENCES `AllowedNames` (`first_name`)"));
    assertThat(
        avroSchema.getProp(SPANNER_FOREIGN_KEY + "1"),
        equalTo(
            "ALTER TABLE `Users` ADD CONSTRAINT `fk_odc` FOREIGN KEY (`last_name`)"
                + " REFERENCES `AllowedNames` (`last_name`) ON DELETE CASCADE"));
    assertThat(
        avroSchema.getProp(SPANNER_FOREIGN_KEY + "2"),
        equalTo(
            "ALTER TABLE `Users` ADD CONSTRAINT `fk_not_enforced_no_action` FOREIGN KEY (`last_name`)"
                + " REFERENCES `AllowedNames` (`last_name`) ON DELETE NO ACTION NOT ENFORCED"));
    assertThat(
        avroSchema.getProp(SPANNER_FOREIGN_KEY + "3"),
        equalTo(
            "ALTER TABLE `Users` ADD CONSTRAINT `fk_enforced` FOREIGN KEY (`last_name`)"
                + " REFERENCES `AllowedNames` (`last_name`) ENFORCED"));
    assertThat(
        avroSchema.getProp(SPANNER_CHECK_CONSTRAINT + "0"),
        equalTo("CONSTRAINT ck CHECK (`first_name` != `last_name`)"));

    System.out.println(avroSchema.toString(true));
  }

  @Test
  public void pgSimple() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder(Dialect.POSTGRESQL)
            .createTable("Users")
            .column("id")
            .pgInt8()
            .notNull()
            .endColumn()
            .column("first_name")
            .pgVarchar()
            .size(10)
            .defaultExpression("'John'")
            .endColumn()
            .column("last_name")
            .type(Type.pgVarchar())
            .max()
            .endColumn()
            .column("full_name")
            .type(Type.pgVarchar())
            .max()
            .generatedAs("CONCAT(first_name, ' ', last_name)")
            .stored()
            .endColumn()
            .column("gen_id")
            .pgInt8()
            .notNull()
            .generatedAs("MOD(id+1, 64)")
            .stored()
            .endColumn()
            .column("gen_id_virtual")
            .pgInt8()
            .notNull()
            .generatedAs("MOD(id+1, 64)")
            .endColumn()
            .column("identity_column")
            .type(Type.int64())
            .isIdentityColumn(true)
            .sequenceKind("bit_reversed_positive")
            .counterStartValue(1000L)
            .skipRangeMin(2000L)
            .skipRangeMax(3000L)
            .endColumn()
            .column("identity_column_no_kind")
            .type(Type.int64())
            .isIdentityColumn(true)
            .counterStartValue(1000L)
            .skipRangeMin(2000L)
            .skipRangeMax(3000L)
            .endColumn()
            .column("tokens")
            .pgSpannerTokenlist()
            .generatedAs("(spanner.tokenize_fulltext(full_name))")
            .isHidden(true)
            .endColumn()
            .column("uuid_column")
            .pgUuid()
            .endColumn()
            .primaryKey()
            .asc("id")
            .asc("gen_id")
            .asc("last_name")
            .end()
            .indexes(
                ImmutableList.of(
                    "CREATE INDEX \"UsersByFirstName\" ON \"Users\" (\"first_name\")",
                    "CREATE SEARCH INDEX \"SearchIndex\" ON \"Users\" (\"tokens\")"
                        + " WITH (sort_order_sharding=TRUE)"))
            .foreignKeys(
                ImmutableList.of(
                    "ALTER TABLE \"Users\" ADD CONSTRAINT \"fk\" FOREIGN KEY (\"first_name\")"
                        + " REFERENCES \"AllowedNames\" (\"first_name\")",
                    "ALTER TABLE \"Users\" ADD CONSTRAINT \"fk_odc\" FOREIGN KEY (\"last_name\")"
                        + " REFERENCES \"AllowedNames\" (\"last_name\") ON DELETE CASCADE"))
            .checkConstraints(
                ImmutableList.of("CONSTRAINT ck CHECK (\"first_name\" != \"last_name\")"))
            .endTable()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroSchema = result.iterator().next();

    assertThat(avroSchema.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));

    assertThat(avroSchema.getName(), equalTo("Users"));

    List<Schema.Field> fields = avroSchema.getFields();

    assertThat(fields, hasSize(10));

    assertThat(fields.get(0).name(), equalTo("id"));
    // Not null
    assertThat(fields.get(0).schema().getType(), equalTo(Schema.Type.LONG));
    assertThat(fields.get(0).getProp(SQL_TYPE), equalTo("bigint"));
    assertThat(fields.get(0).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(0).getProp(GENERATION_EXPRESSION), equalTo(null));
    assertThat(fields.get(0).getProp(STORED), equalTo(null));
    assertThat(fields.get(0).getProp(DEFAULT_EXPRESSION), equalTo(null));

    assertThat(fields.get(1).name(), equalTo("first_name"));
    assertThat(fields.get(1).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(1).getProp(SQL_TYPE), equalTo("character varying(10)"));
    assertThat(fields.get(1).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(1).getProp(GENERATION_EXPRESSION), equalTo(null));
    assertThat(fields.get(1).getProp(STORED), equalTo(null));
    assertThat(fields.get(1).getProp(DEFAULT_EXPRESSION), equalTo("'John'"));

    assertThat(fields.get(2).name(), equalTo("last_name"));
    assertThat(fields.get(2).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(2).getProp(SQL_TYPE), equalTo("character varying"));
    assertThat(fields.get(2).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(2).getProp(GENERATION_EXPRESSION), equalTo(null));
    assertThat(fields.get(2).getProp(STORED), equalTo(null));
    assertThat(fields.get(2).getProp(DEFAULT_EXPRESSION), equalTo(null));

    assertThat(fields.get(3).name(), equalTo("full_name"));
    assertThat(fields.get(3).schema(), equalTo(Schema.create(Schema.Type.NULL)));
    assertThat(fields.get(3).getProp(SQL_TYPE), equalTo("character varying"));
    assertThat(fields.get(3).getProp(NOT_NULL), equalTo("false"));
    assertThat(
        fields.get(3).getProp(GENERATION_EXPRESSION),
        equalTo("CONCAT(first_name, ' ', last_name)"));
    assertThat(fields.get(3).getProp(STORED), equalTo("true"));
    assertThat(fields.get(3).getProp(DEFAULT_EXPRESSION), equalTo(null));

    assertThat(fields.get(4).name(), equalTo("gen_id"));
    assertThat(fields.get(4).schema(), equalTo(Schema.create(Schema.Type.NULL)));
    assertThat(fields.get(4).getProp(SQL_TYPE), equalTo("bigint"));
    assertThat(fields.get(4).getProp(NOT_NULL), equalTo("true"));
    assertThat(fields.get(4).getProp(GENERATION_EXPRESSION), equalTo("MOD(id+1, 64)"));
    assertThat(fields.get(4).getProp(STORED), equalTo("true"));
    assertThat(fields.get(4).getProp(DEFAULT_EXPRESSION), equalTo(null));

    assertThat(fields.get(5).name(), equalTo("gen_id_virtual"));
    assertThat(fields.get(5).schema(), equalTo(Schema.create(Schema.Type.NULL)));
    assertThat(fields.get(5).getProp(SQL_TYPE), equalTo("bigint"));
    assertThat(fields.get(5).getProp(NOT_NULL), equalTo("true"));
    assertThat(fields.get(5).getProp(GENERATION_EXPRESSION), equalTo("MOD(id+1, 64)"));
    assertThat(fields.get(5).getProp(STORED), equalTo("false"));
    assertThat(fields.get(5).getProp(DEFAULT_EXPRESSION), equalTo(null));

    assertThat(fields.get(6).name(), equalTo("identity_column"));
    assertThat(fields.get(6).schema(), equalTo(nullableUnion(Schema.Type.LONG)));
    assertThat(fields.get(6).getProp(SQL_TYPE), equalTo("INT64"));
    assertThat(fields.get(6).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(6).getProp(IDENTITY_COLUMN), equalTo("true"));
    assertThat(fields.get(6).getProp(SPANNER_SEQUENCE_KIND), equalTo("bit_reversed_positive"));
    assertThat(fields.get(6).getProp(SPANNER_SEQUENCE_COUNTER_START), equalTo("1000"));
    assertThat(fields.get(6).getProp(SPANNER_SEQUENCE_SKIP_RANGE_MIN), equalTo("2000"));
    assertThat(fields.get(6).getProp(SPANNER_SEQUENCE_SKIP_RANGE_MAX), equalTo("3000"));

    assertThat(fields.get(7).name(), equalTo("identity_column_no_kind"));
    assertThat(fields.get(7).schema(), equalTo(nullableUnion(Schema.Type.LONG)));
    assertThat(fields.get(7).getProp(SQL_TYPE), equalTo("INT64"));
    assertThat(fields.get(7).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(7).getProp(IDENTITY_COLUMN), equalTo("true"));
    assertThat(fields.get(7).getProp(SPANNER_SEQUENCE_KIND), equalTo(null));
    assertThat(fields.get(7).getProp(SPANNER_SEQUENCE_COUNTER_START), equalTo("1000"));
    assertThat(fields.get(7).getProp(SPANNER_SEQUENCE_SKIP_RANGE_MIN), equalTo("2000"));
    assertThat(fields.get(7).getProp(SPANNER_SEQUENCE_SKIP_RANGE_MAX), equalTo("3000"));

    assertThat(fields.get(8).name(), equalTo("tokens"));
    assertThat(fields.get(8).schema(), equalTo(Schema.create(Schema.Type.NULL)));
    assertThat(fields.get(8).getProp(SQL_TYPE), equalTo("spanner.tokenlist"));
    assertThat(fields.get(8).getProp(NOT_NULL), equalTo("false"));
    assertThat(
        fields.get(8).getProp(GENERATION_EXPRESSION),
        equalTo("(spanner.tokenize_fulltext(full_name))"));
    assertThat(fields.get(8).getProp(HIDDEN), equalTo("true"));
    assertThat(fields.get(8).getProp(DEFAULT_EXPRESSION), equalTo(null));

    assertThat(fields.get(9).name(), equalTo("uuid_column"));
    assertThat(fields.get(9).schema(), equalTo(nullableUuid()));
    assertThat(fields.get(9).getProp(SQL_TYPE), equalTo("uuid"));
    assertThat(fields.get(9).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(9).getProp(GENERATION_EXPRESSION), equalTo(null));
    assertThat(fields.get(9).getProp(STORED), equalTo(null));
    assertThat(fields.get(9).getProp(DEFAULT_EXPRESSION), equalTo(null));

    // spanner pk
    assertThat(avroSchema.getProp(SPANNER_PRIMARY_KEY + "_0"), equalTo("\"id\" ASC"));
    assertThat(avroSchema.getProp(SPANNER_PRIMARY_KEY + "_1"), equalTo("\"gen_id\" ASC"));
    assertThat(avroSchema.getProp(SPANNER_PRIMARY_KEY + "_2"), equalTo("\"last_name\" ASC"));
    assertThat(avroSchema.getProp(SPANNER_PARENT), nullValue());
    assertThat(avroSchema.getProp(SPANNER_ON_DELETE_ACTION), nullValue());

    assertThat(
        avroSchema.getProp(SPANNER_INDEX + "0"),
        equalTo("CREATE INDEX \"UsersByFirstName\" ON \"Users\" (\"first_name\")"));

    assertThat(
        avroSchema.getProp(SPANNER_INDEX + "1"),
        equalTo(
            "CREATE SEARCH INDEX \"SearchIndex\" ON \"Users\" (\"tokens\") WITH (sort_order_sharding=TRUE)"));

    assertThat(
        avroSchema.getProp(SPANNER_FOREIGN_KEY + "0"),
        equalTo(
            "ALTER TABLE \"Users\" ADD CONSTRAINT \"fk\" FOREIGN KEY (\"first_name\")"
                + " REFERENCES \"AllowedNames\" (\"first_name\")"));
    assertThat(
        avroSchema.getProp(SPANNER_FOREIGN_KEY + "1"),
        equalTo(
            "ALTER TABLE \"Users\" ADD CONSTRAINT \"fk_odc\" FOREIGN KEY (\"last_name\")"
                + " REFERENCES \"AllowedNames\" (\"last_name\") ON DELETE CASCADE"));
    assertThat(
        avroSchema.getProp(SPANNER_CHECK_CONSTRAINT + "0"),
        equalTo("CONSTRAINT ck CHECK (\"first_name\" != \"last_name\")"));
  }

  @Test
  public void udfSimple() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder()
            .createUdf("spanner.Foo")
            .dialect(Dialect.GOOGLE_STANDARD_SQL)
            .name("Foo")
            .definition("SELECT 1")
            .endUdf()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroUdf = result.iterator().next();

    assertThat(avroUdf, notNullValue());

    assertThat(avroUdf.getNamespace(), equalTo("spannertest"));
    assertThat(avroUdf.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroUdf.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));
    assertThat(avroUdf.getProp(SPANNER_UDF_NAME), equalTo("Foo"));
    assertThat(avroUdf.getProp(SPANNER_UDF_DEFINITION), equalTo("SELECT 1"));

    assertThat(avroUdf.getName(), equalTo("spanner_Foo"));
  }

  @Test
  public void udfAllOptions() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder()
            .createUdf("spanner.Foo")
            .dialect(Dialect.GOOGLE_STANDARD_SQL)
            .name("Foo")
            .definition("SELECT 1")
            .security(SqlSecurity.INVOKER)
            .type("STRING")
            .addParameter(
                UdfParameter.parse("arg0 STRING", "spanner.Foo", Dialect.GOOGLE_STANDARD_SQL))
            .addParameter(
                UdfParameter.parse(
                    "arg1 STRING DEFAULT \"bar\"", "spanner.Foo", Dialect.GOOGLE_STANDARD_SQL))
            .endUdf()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroUdf = result.iterator().next();

    assertThat(avroUdf, notNullValue());

    assertThat(avroUdf.getNamespace(), equalTo("spannertest"));
    assertThat(avroUdf.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroUdf.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));
    assertThat(avroUdf.getProp(SPANNER_UDF_NAME), equalTo("Foo"));
    assertThat(avroUdf.getProp(SPANNER_UDF_DEFINITION), equalTo("SELECT 1"));
    assertThat(avroUdf.getProp(SPANNER_UDF_SECURITY), equalTo("INVOKER"));
    assertThat(avroUdf.getProp(SPANNER_UDF_TYPE), equalTo("STRING"));
    assertThat(avroUdf.getProp(SPANNER_UDF_PARAMETER + 0), equalTo("`arg0` STRING"));
    assertThat(
        avroUdf.getProp(SPANNER_UDF_PARAMETER + 1), equalTo("`arg1` STRING DEFAULT \"bar\""));

    assertThat(avroUdf.getName(), equalTo("spanner_Foo"));
  }

  @Test
  public void invokerRightsView() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder()
            .createTable("Users")
            .column("id")
            .int64()
            .notNull()
            .endColumn()
            .column("first_name")
            .string()
            .size(10)
            .endColumn()
            .column("last_name")
            .type(Type.string())
            .max()
            .endColumn()
            .endTable()
            .createView("Names")
            .query("SELECT first_name, last_name FROM Users")
            .security(View.SqlSecurity.INVOKER)
            .endView()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(2));
    Schema avroView = null;
    for (Schema s : result) {
      if (s.getName().equals("Names")) {
        avroView = s;
      }
    }
    assertThat(avroView, notNullValue());

    assertThat(avroView.getNamespace(), equalTo("spannertest"));
    assertThat(avroView.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroView.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));
    assertThat(
        avroView.getProp(SPANNER_VIEW_QUERY), equalTo("SELECT first_name, last_name FROM Users"));
    assertThat(avroView.getProp(SPANNER_VIEW_SECURITY), equalTo("INVOKER"));

    assertThat(avroView.getName(), equalTo("Names"));
  }

  @Test
  public void pgInvokerRightsView() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder(Dialect.POSTGRESQL)
            .createTable("Users")
            .column("id")
            .pgInt8()
            .notNull()
            .endColumn()
            .column("first_name")
            .pgVarchar()
            .size(10)
            .endColumn()
            .column("last_name")
            .type(Type.pgVarchar())
            .max()
            .endColumn()
            .endTable()
            .createView("Names")
            .query("SELECT first_name, last_name FROM Users")
            .security(View.SqlSecurity.INVOKER)
            .endView()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(2));
    Schema avroView = null;
    for (Schema s : result) {
      if (s.getName().equals("Names")) {
        avroView = s;
      }
    }
    assertThat(avroView, notNullValue());

    assertThat(avroView.getNamespace(), equalTo("spannertest"));
    assertThat(avroView.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroView.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));
    assertThat(
        avroView.getProp(SPANNER_VIEW_QUERY), equalTo("SELECT first_name, last_name FROM Users"));
    assertThat(avroView.getProp(SPANNER_VIEW_SECURITY), equalTo("INVOKER"));

    assertThat(avroView.getName(), equalTo("Names"));
  }

  @Test
  public void definerRightsView() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder()
            .createTable("Users")
            .column("id")
            .int64()
            .notNull()
            .endColumn()
            .column("first_name")
            .string()
            .size(10)
            .endColumn()
            .column("last_name")
            .type(Type.string())
            .max()
            .endColumn()
            .endTable()
            .createView("Names")
            .query("SELECT first_name, last_name FROM Users")
            .security(View.SqlSecurity.DEFINER)
            .endView()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(2));
    Schema avroView = null;
    for (Schema s : result) {
      if (s.getName().equals("Names")) {
        avroView = s;
      }
    }
    assertThat(avroView, notNullValue());

    assertThat(avroView.getNamespace(), equalTo("spannertest"));
    assertThat(avroView.getProp("googleFormatVersion"), equalTo("booleans"));
    assertThat(avroView.getProp("googleStorage"), equalTo("CloudSpanner"));
    assertThat(
        avroView.getProp("spannerViewQuery"), equalTo("SELECT first_name, last_name FROM Users"));
    assertThat(avroView.getProp("spannerViewSecurity"), equalTo("DEFINER"));

    assertThat(avroView.getName(), equalTo("Names"));
  }

  @Test
  public void pgDefinerRightsView() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder(Dialect.POSTGRESQL)
            .createTable("Users")
            .column("id")
            .pgInt8()
            .notNull()
            .endColumn()
            .column("first_name")
            .pgVarchar()
            .size(10)
            .endColumn()
            .column("last_name")
            .type(Type.pgVarchar())
            .max()
            .endColumn()
            .endTable()
            .createView("Names")
            .query("SELECT first_name, last_name FROM Users")
            .security(View.SqlSecurity.DEFINER)
            .endView()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(2));
    Schema avroView = null;
    for (Schema s : result) {
      if (s.getName().equals("Names")) {
        avroView = s;
      }
    }
    assertThat(avroView, notNullValue());

    assertThat(avroView.getNamespace(), equalTo("spannertest"));
    assertThat(avroView.getProp("googleFormatVersion"), equalTo("booleans"));
    assertThat(avroView.getProp("googleStorage"), equalTo("CloudSpanner"));
    assertThat(
        avroView.getProp("spannerViewQuery"), equalTo("SELECT first_name, last_name FROM Users"));
    assertThat(avroView.getProp("spannerViewSecurity"), equalTo("DEFINER"));

    assertThat(avroView.getName(), equalTo("Names"));
  }

  @Test
  public void allTypes() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder()
            .createTable("AllTYPES")
            .column("bool_field")
            .bool()
            .endColumn()
            .column("int64_field")
            .int64()
            .endColumn()
            .column("float64_field")
            .float64()
            .endColumn()
            .column("string_field")
            .string()
            .max()
            .endColumn()
            .column("bytes_field")
            .bytes()
            .max()
            .endColumn()
            .column("timestamp_field")
            .timestamp()
            .endColumn()
            .column("date_field")
            .date()
            .endColumn()
            .column("numeric_field")
            .numeric()
            .endColumn()
            .column("json_field")
            .json()
            .endColumn()
            .column("arr_bool_field")
            .type(Type.array(Type.bool()))
            .endColumn()
            .column("arr_int64_field")
            .type(Type.array(Type.int64()))
            .endColumn()
            .column("arr_float64_field")
            .type(Type.array(Type.float64()))
            .endColumn()
            .column("arr_string_field")
            .type(Type.array(Type.string()))
            .max()
            .endColumn()
            .column("arr_bytes_field")
            .type(Type.array(Type.bytes()))
            .max()
            .endColumn()
            .column("arr_timestamp_field")
            .type(Type.array(Type.timestamp()))
            .endColumn()
            .column("arr_date_field")
            .type(Type.array(Type.date()))
            .endColumn()
            .column("arr_numeric_field")
            .type(Type.array(Type.numeric()))
            .endColumn()
            .column("arr_json_field")
            .type(Type.array(Type.json()))
            .endColumn()
            .column("proto_field")
            .type(Type.proto("com.google.cloud.teleport.spanner.tests.TestMessage"))
            .endColumn()
            .column("arr_proto_field")
            .type(Type.array(Type.proto("com.google.cloud.teleport.spanner.tests.TestMessage")))
            .endColumn()
            .column("enum_field")
            .type(Type.protoEnum("com.google.cloud.teleport.spanner.tests.TestEnum"))
            .endColumn()
            .column("arr_enum_field")
            .type(Type.array(Type.protoEnum("com.google.cloud.teleport.spanner.tests.TestEnum")))
            .endColumn()
            .column("uuid_col")
            .type(Type.uuid())
            .endColumn()
            .column("uuid_array_col")
            .type(Type.array(Type.uuid()))
            .endColumn()
            .primaryKey()
            .asc("bool_field")
            .end()
            .interleaveInParent("ParentTable")
            .onDeleteCascade()
            .endTable()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroSchema = result.iterator().next();

    assertThat(avroSchema.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));

    List<Schema.Field> fields = avroSchema.getFields();

    assertThat(fields, hasSize(24));

    assertThat(fields.get(0).name(), equalTo("bool_field"));
    assertThat(fields.get(0).schema(), equalTo(nullableUnion(Schema.Type.BOOLEAN)));
    assertThat(fields.get(0).getProp(SQL_TYPE), equalTo("BOOL"));

    assertThat(fields.get(1).name(), equalTo("int64_field"));
    assertThat(fields.get(1).schema(), equalTo(nullableUnion(Schema.Type.LONG)));
    assertThat(fields.get(1).getProp(SQL_TYPE), equalTo("INT64"));

    assertThat(fields.get(2).name(), equalTo("float64_field"));
    assertThat(fields.get(2).schema(), equalTo(nullableUnion(Schema.Type.DOUBLE)));
    assertThat(fields.get(2).getProp(SQL_TYPE), equalTo("FLOAT64"));

    assertThat(fields.get(3).name(), equalTo("string_field"));
    assertThat(fields.get(3).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(3).getProp(SQL_TYPE), equalTo("STRING(MAX)"));

    assertThat(fields.get(4).name(), equalTo("bytes_field"));
    assertThat(fields.get(4).schema(), equalTo(nullableUnion(Schema.Type.BYTES)));
    assertThat(fields.get(4).getProp(SQL_TYPE), equalTo("BYTES(MAX)"));

    assertThat(fields.get(5).name(), equalTo("timestamp_field"));
    assertThat(fields.get(5).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(5).getProp(SQL_TYPE), equalTo("TIMESTAMP"));

    assertThat(fields.get(6).name(), equalTo("date_field"));
    assertThat(fields.get(6).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(6).getProp(SQL_TYPE), equalTo("DATE"));

    assertThat(fields.get(7).name(), equalTo("numeric_field"));
    assertThat(fields.get(7).schema(), equalTo(nullableNumericUnion()));
    assertThat(fields.get(7).getProp(SQL_TYPE), equalTo("NUMERIC"));

    assertThat(fields.get(8).name(), equalTo("json_field"));
    assertThat(fields.get(8).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(8).getProp(SQL_TYPE), equalTo("JSON"));

    assertThat(fields.get(9).name(), equalTo("arr_bool_field"));
    assertThat(fields.get(9).schema(), equalTo(nullableArray(Schema.Type.BOOLEAN)));
    assertThat(fields.get(9).getProp(SQL_TYPE), equalTo("ARRAY<BOOL>"));

    assertThat(fields.get(10).name(), equalTo("arr_int64_field"));
    assertThat(fields.get(10).schema(), equalTo(nullableArray(Schema.Type.LONG)));
    assertThat(fields.get(10).getProp(SQL_TYPE), equalTo("ARRAY<INT64>"));

    assertThat(fields.get(11).name(), equalTo("arr_float64_field"));
    assertThat(fields.get(11).schema(), equalTo(nullableArray(Schema.Type.DOUBLE)));
    assertThat(fields.get(11).getProp(SQL_TYPE), equalTo("ARRAY<FLOAT64>"));

    assertThat(fields.get(12).name(), equalTo("arr_string_field"));
    assertThat(fields.get(12).schema(), equalTo(nullableArray(Schema.Type.STRING)));
    assertThat(fields.get(12).getProp(SQL_TYPE), equalTo("ARRAY<STRING(MAX)>"));

    assertThat(fields.get(13).name(), equalTo("arr_bytes_field"));
    assertThat(fields.get(13).schema(), equalTo(nullableArray(Schema.Type.BYTES)));
    assertThat(fields.get(13).getProp(SQL_TYPE), equalTo("ARRAY<BYTES(MAX)>"));

    assertThat(fields.get(14).name(), equalTo("arr_timestamp_field"));
    assertThat(fields.get(14).schema(), equalTo(nullableArray(Schema.Type.STRING)));
    assertThat(fields.get(14).getProp(SQL_TYPE), equalTo("ARRAY<TIMESTAMP>"));

    assertThat(fields.get(15).name(), equalTo("arr_date_field"));
    assertThat(fields.get(15).schema(), equalTo(nullableArray(Schema.Type.STRING)));
    assertThat(fields.get(15).getProp(SQL_TYPE), equalTo("ARRAY<DATE>"));

    assertThat(fields.get(16).name(), equalTo("arr_numeric_field"));
    assertThat(fields.get(16).schema(), equalTo(nullableNumericArray()));
    assertThat(fields.get(16).getProp(SQL_TYPE), equalTo("ARRAY<NUMERIC>"));

    assertThat(fields.get(17).name(), equalTo("arr_json_field"));
    assertThat(fields.get(17).schema(), equalTo(nullableArray(Schema.Type.STRING)));
    assertThat(fields.get(17).getProp(SQL_TYPE), equalTo("ARRAY<JSON>"));

    assertThat(fields.get(18).name(), equalTo("proto_field"));
    assertThat(fields.get(18).schema(), equalTo(nullableUnion(Schema.Type.BYTES)));
    assertThat(
        fields.get(18).getProp(SQL_TYPE),
        equalTo("PROTO<com.google.cloud.teleport.spanner.tests.TestMessage>"));

    assertThat(fields.get(19).name(), equalTo("arr_proto_field"));
    assertThat(fields.get(19).schema(), equalTo(nullableArray(Schema.Type.BYTES)));
    assertThat(
        fields.get(19).getProp(SQL_TYPE),
        equalTo("ARRAY<PROTO<com.google.cloud.teleport.spanner.tests.TestMessage>>"));

    assertThat(fields.get(20).name(), equalTo("enum_field"));
    assertThat(fields.get(20).schema(), equalTo(nullableUnion(Schema.Type.LONG)));
    assertThat(
        fields.get(20).getProp(SQL_TYPE),
        equalTo("ENUM<com.google.cloud.teleport.spanner.tests.TestEnum>"));

    assertThat(fields.get(21).name(), equalTo("arr_enum_field"));
    assertThat(fields.get(21).schema(), equalTo(nullableArray(Schema.Type.LONG)));
    assertThat(
        fields.get(21).getProp(SQL_TYPE),
        equalTo("ARRAY<ENUM<com.google.cloud.teleport.spanner.tests.TestEnum>>"));

    Schema.Field field22 = fields.get(22);
    assertThat(field22.name(), equalTo("uuid_col"));
    assertThat(field22.schema(), equalTo(nullableUuid()));
    assertThat(field22.getProp(SQL_TYPE), equalTo("UUID"));

    Schema.Field field23 = fields.get(23);
    assertThat(field23.name(), equalTo("uuid_array_col"));
    assertThat(field23.schema(), equalTo(nullableUuidArray()));
    assertThat(field23.getProp(SQL_TYPE), equalTo("ARRAY<UUID>"));

    assertThat(avroSchema.getProp(SPANNER_PRIMARY_KEY + "_0"), equalTo("`bool_field` ASC"));
    assertThat(avroSchema.getProp(SPANNER_PARENT), equalTo("ParentTable"));
    assertThat(avroSchema.getProp(SPANNER_ON_DELETE_ACTION), equalTo("cascade"));
    assertThat(avroSchema.getProp(SPANNER_INTERLEAVE_TYPE), equalTo("IN PARENT"));

    System.out.println(avroSchema.toString(true));
  }

  @Test
  public void pgAllTypes() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder(Dialect.POSTGRESQL)
            .createTable("AllTYPES")
            .column("bool_field")
            .pgBool()
            .endColumn()
            .column("int8_field")
            .pgInt8()
            .endColumn()
            .column("float8_field")
            .pgFloat8()
            .endColumn()
            .column("varchar_field")
            .pgVarchar()
            .max()
            .endColumn()
            .column("bytea_field")
            .pgBytea()
            .endColumn()
            .column("timestamptz_field")
            .pgTimestamptz()
            .endColumn()
            .column("numeric_field")
            .pgNumeric()
            .endColumn()
            .column("text_field")
            .pgText()
            .endColumn()
            .column("date_field")
            .pgDate()
            .endColumn()
            .column("commit_timestamp_field")
            .pgSpannerCommitTimestamp()
            .endColumn()
            .column("arr_bool_field")
            .type(Type.pgArray(Type.pgBool()))
            .endColumn()
            .column("arr_int8_field")
            .type(Type.pgArray(Type.pgInt8()))
            .endColumn()
            .column("arr_float8_field")
            .type(Type.pgArray(Type.pgFloat8()))
            .endColumn()
            .column("arr_varchar_field")
            .type(Type.pgArray(Type.pgVarchar()))
            .max()
            .endColumn()
            .column("arr_bytea_field")
            .type(Type.pgArray(Type.pgBytea()))
            .endColumn()
            .column("arr_timestamptz_field")
            .type(Type.pgArray(Type.pgTimestamptz()))
            .endColumn()
            .column("arr_numeric_field")
            .type(Type.pgArray(Type.pgNumeric()))
            .endColumn()
            .column("arr_text_field")
            .type(Type.pgArray(Type.pgText()))
            .endColumn()
            .column("arr_date_field")
            .type(Type.pgArray(Type.pgDate()))
            .endColumn()
            .column("uuid_col")
            .type(Type.pgUuid())
            .endColumn()
            .column("uuid_array_col")
            .type(Type.pgArray(Type.pgUuid()))
            .endColumn()
            .primaryKey()
            .asc("bool_field")
            .end()
            .interleaveInParent("ParentTable")
            .onDeleteCascade()
            .endTable()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroSchema = result.iterator().next();

    assertThat(avroSchema.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));

    List<Schema.Field> fields = avroSchema.getFields();

    assertThat(fields, hasSize(21));

    assertThat(fields.get(0).name(), equalTo("bool_field"));
    assertThat(fields.get(0).schema(), equalTo(nullableUnion(Schema.Type.BOOLEAN)));
    assertThat(fields.get(0).getProp(SQL_TYPE), equalTo("boolean"));

    assertThat(fields.get(1).name(), equalTo("int8_field"));
    assertThat(fields.get(1).schema(), equalTo(nullableUnion(Schema.Type.LONG)));
    assertThat(fields.get(1).getProp(SQL_TYPE), equalTo("bigint"));

    assertThat(fields.get(2).name(), equalTo("float8_field"));
    assertThat(fields.get(2).schema(), equalTo(nullableUnion(Schema.Type.DOUBLE)));
    assertThat(fields.get(2).getProp(SQL_TYPE), equalTo("double precision"));

    assertThat(fields.get(3).name(), equalTo("varchar_field"));
    assertThat(fields.get(3).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(3).getProp(SQL_TYPE), equalTo("character varying"));

    assertThat(fields.get(4).name(), equalTo("bytea_field"));
    assertThat(fields.get(4).schema(), equalTo(nullableUnion(Schema.Type.BYTES)));
    assertThat(fields.get(4).getProp(SQL_TYPE), equalTo("bytea"));

    assertThat(fields.get(5).name(), equalTo("timestamptz_field"));
    assertThat(fields.get(5).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(5).getProp(SQL_TYPE), equalTo("timestamp with time zone"));

    assertThat(fields.get(6).name(), equalTo("numeric_field"));
    assertThat(fields.get(6).schema(), equalTo(nullablePgNumericUnion()));
    assertThat(fields.get(6).getProp(SQL_TYPE), equalTo("numeric"));

    assertThat(fields.get(7).name(), equalTo("text_field"));
    assertThat(fields.get(7).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(7).getProp(SQL_TYPE), equalTo("text"));

    assertThat(fields.get(8).name(), equalTo("date_field"));
    assertThat(fields.get(8).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(8).getProp(SQL_TYPE), equalTo("date"));

    assertThat(fields.get(9).name(), equalTo("commit_timestamp_field"));
    assertThat(fields.get(9).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(9).getProp(SQL_TYPE), equalTo("spanner.commit_timestamp"));

    assertThat(fields.get(10).name(), equalTo("arr_bool_field"));
    assertThat(fields.get(10).schema(), equalTo(nullableArray(Schema.Type.BOOLEAN)));
    assertThat(fields.get(10).getProp(SQL_TYPE), equalTo("boolean[]"));

    assertThat(fields.get(11).name(), equalTo("arr_int8_field"));
    assertThat(fields.get(11).schema(), equalTo(nullableArray(Schema.Type.LONG)));
    assertThat(fields.get(11).getProp(SQL_TYPE), equalTo("bigint[]"));

    assertThat(fields.get(12).name(), equalTo("arr_float8_field"));
    assertThat(fields.get(12).schema(), equalTo(nullableArray(Schema.Type.DOUBLE)));
    assertThat(fields.get(12).getProp(SQL_TYPE), equalTo("double precision[]"));

    assertThat(fields.get(13).name(), equalTo("arr_varchar_field"));
    assertThat(fields.get(13).schema(), equalTo(nullableArray(Schema.Type.STRING)));
    assertThat(fields.get(13).getProp(SQL_TYPE), equalTo("character varying[]"));

    assertThat(fields.get(14).name(), equalTo("arr_bytea_field"));
    assertThat(fields.get(14).schema(), equalTo(nullableArray(Schema.Type.BYTES)));
    assertThat(fields.get(14).getProp(SQL_TYPE), equalTo("bytea[]"));

    assertThat(fields.get(15).name(), equalTo("arr_timestamptz_field"));
    assertThat(fields.get(15).schema(), equalTo(nullableArray(Schema.Type.STRING)));
    assertThat(fields.get(15).getProp(SQL_TYPE), equalTo("timestamp with time zone[]"));

    assertThat(fields.get(16).name(), equalTo("arr_numeric_field"));
    assertThat(fields.get(16).schema(), equalTo(nullablePgNumericArray()));
    assertThat(fields.get(16).getProp(SQL_TYPE), equalTo("numeric[]"));

    assertThat(fields.get(17).name(), equalTo("arr_text_field"));
    assertThat(fields.get(17).schema(), equalTo(nullableArray(Schema.Type.STRING)));
    assertThat(fields.get(17).getProp(SQL_TYPE), equalTo("text[]"));

    assertThat(fields.get(18).name(), equalTo("arr_date_field"));
    assertThat(fields.get(18).schema(), equalTo(nullableArray(Schema.Type.STRING)));
    assertThat(fields.get(18).getProp(SQL_TYPE), equalTo("date[]"));

    Schema.Field field19 = fields.get(19);
    assertThat(field19.name(), equalTo("uuid_col"));
    assertThat(field19.schema(), equalTo(nullableUuid()));
    assertThat(field19.getProp(SQL_TYPE), equalTo("uuid"));

    Schema.Field field20 = fields.get(20);
    assertThat(field20.name(), equalTo("uuid_array_col"));
    assertThat(field20.schema(), equalTo(nullableUuidArray()));
    assertThat(field20.getProp(SQL_TYPE), equalTo("uuid[]"));

    assertThat(avroSchema.getProp(SPANNER_PRIMARY_KEY + "_0"), equalTo("\"bool_field\" ASC"));
    assertThat(avroSchema.getProp(SPANNER_PARENT), equalTo("ParentTable"));
    assertThat(avroSchema.getProp(SPANNER_ON_DELETE_ACTION), equalTo("cascade"));
    assertThat(avroSchema.getProp(SPANNER_INTERLEAVE_TYPE), equalTo("IN PARENT"));
  }

  @Test
  public void timestampLogicalTypeTest() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", true);
    Ddl ddl =
        Ddl.builder()
            .createTable("Users")
            .column("id")
            .int64()
            .notNull()
            .endColumn()
            .column("timestamp_field")
            .timestamp()
            .endColumn()
            .primaryKey()
            .asc("id")
            .end()
            .endTable()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroSchema = result.iterator().next();

    assertThat(avroSchema.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));

    assertThat(avroSchema.getName(), equalTo("Users"));

    List<Schema.Field> fields = avroSchema.getFields();

    assertThat(fields, hasSize(2));

    assertThat(fields.get(0).name(), equalTo("id"));
    // Not null
    assertThat(fields.get(0).schema().getType(), equalTo(Schema.Type.LONG));
    assertThat(fields.get(0).getProp(SQL_TYPE), equalTo("INT64"));
    assertThat(fields.get(0).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(0).getProp(GENERATION_EXPRESSION), equalTo(null));
    assertThat(fields.get(0).getProp(STORED), equalTo(null));

    assertThat(fields.get(1).name(), equalTo("timestamp_field"));
    assertThat(fields.get(1).schema(), equalTo(nullableTimestampUnion()));
    assertThat(fields.get(1).getProp(SQL_TYPE), equalTo("TIMESTAMP"));
  }

  @Test
  public void pgTimestampLogicalTypeTest() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", true);
    Ddl ddl =
        Ddl.builder(Dialect.POSTGRESQL)
            .createTable("Users")
            .column("id")
            .pgInt8()
            .notNull()
            .endColumn()
            .column("timestamp_field")
            .pgTimestamptz()
            .endColumn()
            .primaryKey()
            .asc("id")
            .end()
            .endTable()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroSchema = result.iterator().next();

    assertThat(avroSchema.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));

    assertThat(avroSchema.getName(), equalTo("Users"));

    List<Schema.Field> fields = avroSchema.getFields();

    assertThat(fields, hasSize(2));

    assertThat(fields.get(0).name(), equalTo("id"));
    // Not null
    assertThat(fields.get(0).schema().getType(), equalTo(Schema.Type.LONG));
    assertThat(fields.get(0).getProp(SQL_TYPE), equalTo("bigint"));
    assertThat(fields.get(0).getProp(NOT_NULL), equalTo(null));
    assertThat(fields.get(0).getProp(GENERATION_EXPRESSION), equalTo(null));
    assertThat(fields.get(0).getProp(STORED), equalTo(null));

    assertThat(fields.get(1).name(), equalTo("timestamp_field"));
    assertThat(fields.get(1).schema(), equalTo(nullableTimestampUnion()));
    assertThat(fields.get(1).getProp(SQL_TYPE), equalTo("timestamp with time zone"));
  }

  @Test
  public void propertyGraphs() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", true);

    // Craft Property Declarations
    PropertyGraph.PropertyDeclaration propertyDeclaration1 =
        new PropertyGraph.PropertyDeclaration("dummyPropName", "dummyPropType");
    PropertyGraph.PropertyDeclaration propertyDeclaration2 =
        new PropertyGraph.PropertyDeclaration("aliasedPropName", "aliasedPropType");
    ImmutableList<String> propertyDeclsLabel1 =
        ImmutableList.copyOf(Arrays.asList(propertyDeclaration1.name, propertyDeclaration2.name));

    // Craft Labels and associated property definitions
    PropertyGraph.GraphElementLabel label1 =
        new PropertyGraph.GraphElementLabel("dummyLabelName1", propertyDeclsLabel1);
    GraphElementTable.PropertyDefinition propertyDefinition1 =
        new PropertyDefinition("dummyPropName", "dummyPropName");
    GraphElementTable.PropertyDefinition propertyDefinition2 =
        new PropertyDefinition(
            "aliasedPropName", "CONCAT(CAST(test_col AS STRING), \":\", \"dummyColumn\")");
    GraphElementTable.LabelToPropertyDefinitions labelToPropertyDefinitions1 =
        new LabelToPropertyDefinitions(
            label1.name, ImmutableList.of(propertyDefinition1, propertyDefinition2));

    PropertyGraph.GraphElementLabel label2 =
        new PropertyGraph.GraphElementLabel("dummyLabelName2", ImmutableList.of());
    GraphElementTable.LabelToPropertyDefinitions labelToPropertyDefinitions2 =
        new LabelToPropertyDefinitions(label2.name, ImmutableList.of());

    PropertyGraph.GraphElementLabel label3 =
        new PropertyGraph.GraphElementLabel("dummyLabelName3", ImmutableList.of());
    GraphElementTable.LabelToPropertyDefinitions labelToPropertyDefinitions3 =
        new LabelToPropertyDefinitions(label3.name, ImmutableList.of());

    // Craft Node table
    GraphElementTable.Builder testNodeTable =
        GraphElementTable.builder()
            .baseTableName("baseTable")
            .name("nodeAlias")
            .kind(GraphElementTable.Kind.NODE)
            .keyColumns(ImmutableList.of("primaryKey"))
            .labelToPropertyDefinitions(
                ImmutableList.of(labelToPropertyDefinitions1, labelToPropertyDefinitions2));

    // Craft Edge table
    GraphElementTable.Builder testEdgeTable =
        GraphElementTable.builder()
            .baseTableName("edgeBaseTable")
            .name("edgeAlias")
            .kind(GraphElementTable.Kind.EDGE)
            .keyColumns(ImmutableList.of("edgePrimaryKey"))
            .sourceNodeTable(
                new GraphNodeTableReference(
                    "baseTable", ImmutableList.of("nodeKey"), ImmutableList.of("sourceEdgeKey")))
            .targetNodeTable(
                new GraphNodeTableReference(
                    "baseTable", ImmutableList.of("otherNodeKey"), ImmutableList.of("destEdgeKey")))
            .labelToPropertyDefinitions(ImmutableList.of(labelToPropertyDefinitions3));

    Ddl ddl =
        Ddl.builder()
            .createPropertyGraph("testGraph")
            .addLabel(label1)
            .addLabel(label2)
            .addLabel(label3)
            .addPropertyDeclaration(propertyDeclaration1)
            .addPropertyDeclaration(propertyDeclaration2)
            .addNodeTable(testNodeTable.autoBuild())
            .addEdgeTable(testEdgeTable.autoBuild())
            .endPropertyGraph()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroSchema = result.iterator().next();

    assertThat(avroSchema.getName(), equalTo("testGraph"));
    assertThat(avroSchema.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));
    assertThat(avroSchema.getProp(SPANNER_ENTITY), equalTo(SPANNER_ENTITY_PROPERTY_GRAPH));

    // Basic assertions
    assertEquals("testGraph", avroSchema.getName());
    assertEquals("spannertest", avroSchema.getNamespace());
    assertEquals(Schema.Type.RECORD, avroSchema.getType());

    // Asserting properties related to Spanner
    assertEquals("CloudSpanner", avroSchema.getProp(GOOGLE_STORAGE));
    assertEquals("testGraph", avroSchema.getProp("spannerName"));
    assertEquals("booleans", avroSchema.getProp(GOOGLE_FORMAT_VERSION));

    // Asserting properties related to Node table
    assertEquals("nodeAlias", avroSchema.getProp(SPANNER_NODE_TABLE + "_0_NAME"));
    assertEquals("baseTable", avroSchema.getProp(SPANNER_NODE_TABLE + "_0_BASE_TABLE_NAME"));
    assertEquals("NODE", avroSchema.getProp(SPANNER_NODE_TABLE + "_0_KIND"));
    assertEquals("primaryKey", avroSchema.getProp(SPANNER_NODE_TABLE + "_0_KEY_COLUMNS"));

    // Asserting properties related to Node labels
    assertEquals("dummyLabelName1", avroSchema.getProp(SPANNER_NODE_TABLE + "_0_LABEL_0_NAME"));
    assertEquals("dummyLabelName2", avroSchema.getProp(SPANNER_NODE_TABLE + "_0_LABEL_1_NAME"));

    // Asserting properties related to Node label properties
    assertEquals(
        "dummyPropName", avroSchema.getProp(SPANNER_NODE_TABLE + "_0_LABEL_0_PROPERTY_0_NAME"));
    assertEquals(
        "dummyPropName", avroSchema.getProp(SPANNER_NODE_TABLE + "_0_LABEL_0_PROPERTY_0_VALUE"));
    assertEquals(
        "aliasedPropName", avroSchema.getProp(SPANNER_NODE_TABLE + "_0_LABEL_0_PROPERTY_1_NAME"));
    assertEquals(
        "CONCAT(CAST(test_col AS STRING), \":\", \"dummyColumn\")",
        avroSchema.getProp(SPANNER_NODE_TABLE + "_0_LABEL_0_PROPERTY_1_VALUE"));

    // Asserting properties related to Edge table
    assertEquals("edgeAlias", avroSchema.getProp(SPANNER_EDGE_TABLE + "_0_NAME"));
    assertEquals("edgeBaseTable", avroSchema.getProp(SPANNER_EDGE_TABLE + "_0_BASE_TABLE_NAME"));
    assertEquals("EDGE", avroSchema.getProp(SPANNER_EDGE_TABLE + "_0_KIND"));
    assertEquals("edgePrimaryKey", avroSchema.getProp(SPANNER_EDGE_TABLE + "_0_KEY_COLUMNS"));
    assertEquals("baseTable", avroSchema.getProp(SPANNER_EDGE_TABLE + "_0_SOURCE_NODE_TABLE_NAME"));
    assertEquals("nodeKey", avroSchema.getProp(SPANNER_EDGE_TABLE + "_0_SOURCE_NODE_KEY_COLUMNS"));
    assertEquals(
        "sourceEdgeKey", avroSchema.getProp(SPANNER_EDGE_TABLE + "_0_SOURCE_EDGE_KEY_COLUMNS"));
    assertEquals("baseTable", avroSchema.getProp(SPANNER_EDGE_TABLE + "_0_TARGET_NODE_TABLE_NAME"));
    assertEquals(
        "otherNodeKey", avroSchema.getProp(SPANNER_EDGE_TABLE + "_0_TARGET_NODE_KEY_COLUMNS"));
    assertEquals(
        "destEdgeKey", avroSchema.getProp(SPANNER_EDGE_TABLE + "_0_TARGET_EDGE_KEY_COLUMNS"));

    // Asserting properties related to Edge labels
    assertEquals("dummyLabelName3", avroSchema.getProp(SPANNER_EDGE_TABLE + "_0_LABEL_0_NAME"));

    // Asserting labels and properties linked to them
    assertEquals("dummyLabelName1", avroSchema.getProp(SPANNER_LABEL + "_0_NAME"));
    assertEquals("dummyPropName", avroSchema.getProp(SPANNER_LABEL + "_0_PROPERTY_0"));
    assertEquals("aliasedPropName", avroSchema.getProp(SPANNER_LABEL + "_0_PROPERTY_1"));

    assertEquals("dummyLabelName2", avroSchema.getProp(SPANNER_LABEL + "_1_NAME"));
    assertEquals("dummyLabelName3", avroSchema.getProp(SPANNER_LABEL + "_2_NAME"));

    // Asserting properties related to graph property declarations
    assertEquals("dummyPropName", avroSchema.getProp(SPANNER_PROPERTY_DECLARATION + "_0_NAME"));
    assertEquals("dummyPropType", avroSchema.getProp(SPANNER_PROPERTY_DECLARATION + "_0_TYPE"));
    assertEquals("aliasedPropName", avroSchema.getProp(SPANNER_PROPERTY_DECLARATION + "_1_NAME"));
    assertEquals("aliasedPropType", avroSchema.getProp(SPANNER_PROPERTY_DECLARATION + "_1_TYPE"));
  }

  @Test
  public void models() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", true);
    Ddl ddl =
        Ddl.builder()
            .createModel("ModelAll")
            .inputColumn("i1")
            .type(Type.bool())
            .size(-1)
            .columnOptions(ImmutableList.of("required=FALSE"))
            .endInputColumn()
            .inputColumn("i2")
            .type(Type.string())
            .size(-1)
            .endInputColumn()
            .outputColumn("o1")
            .type(Type.int64())
            .size(-1)
            .columnOptions(ImmutableList.of("required=TRUE"))
            .endOutputColumn()
            .outputColumn("o2")
            .type(Type.float64())
            .size(-1)
            .endOutputColumn()
            .remote(true)
            .options(ImmutableList.of("endpoint=\"test\""))
            .endModel()
            .createModel("ModelMin")
            .remote(false)
            .inputColumn("i1")
            .type(Type.bool())
            .size(-1)
            .endInputColumn()
            .outputColumn("o1")
            .type(Type.int64())
            .size(-1)
            .endOutputColumn()
            .endModel()
            .createModel("ModelStruct")
            .inputColumn("i1")
            .type(Type.struct(StructField.of("a", Type.bool())))
            .size(-1)
            .endInputColumn()
            .outputColumn("o1")
            .type(
                Type.struct(
                    ImmutableList.of(
                        StructField.of("a", Type.bool()),
                        StructField.of(
                            "b",
                            Type.array(
                                Type.struct(
                                    ImmutableList.of(
                                        StructField.of("c", Type.string()),
                                        StructField.of("d", Type.array(Type.float64())))))),
                        StructField.of(
                            "e",
                            Type.struct(
                                ImmutableList.of(
                                    StructField.of(
                                        "f",
                                        Type.struct(
                                            ImmutableList.of(
                                                StructField.of("g", Type.int64()))))))))))
            .size(-1)
            .endOutputColumn()
            .remote(false)
            .endModel()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(3));

    Iterator<Schema> iterator = result.iterator();
    Schema s = iterator.next();
    assertThat(s.getName(), equalTo("ModelAll"));
    assertThat(s.getNamespace(), equalTo("spannertest"));
    assertThat(s.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(s.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));
    assertThat(s.getProp(SPANNER_ENTITY), equalTo(SPANNER_ENTITY_MODEL));
    assertThat(s.getProp(SPANNER_REMOTE), equalTo("true"));
    assertThat(s.getProp(SPANNER_OPTION + "0"), equalTo("endpoint=\"test\""));
    assertThat(s.getFields(), hasSize(2));
    assertThat(s.getFields().get(0).name(), equalTo(INPUT));
    assertThat(s.getFields().get(0).schema().getType(), equalTo(Schema.Type.RECORD));
    assertThat(s.getFields().get(0).schema().getName(), equalTo("ModelAll_Input"));
    assertThat(s.getFields().get(0).schema().getFields(), hasSize(2));
    assertThat(s.getFields().get(0).schema().getFields().get(0).name(), equalTo("i1"));
    assertThat(
        s.getFields().get(0).schema().getFields().get(0).schema().getType(),
        equalTo(Schema.Type.BOOLEAN));
    assertThat(s.getFields().get(0).schema().getFields().get(0).getProp(SQL_TYPE), equalTo("BOOL"));
    assertThat(
        s.getFields().get(0).schema().getFields().get(0).getProp(SPANNER_OPTION + "0"),
        equalTo("required=FALSE"));
    assertThat(s.getFields().get(0).schema().getFields().get(1).name(), equalTo("i2"));
    assertThat(
        s.getFields().get(0).schema().getFields().get(1).schema().getType(),
        equalTo(Schema.Type.STRING));
    assertThat(
        s.getFields().get(0).schema().getFields().get(1).getProp(SQL_TYPE), equalTo("STRING(MAX)"));
    assertThat(s.getFields().get(1).name(), equalTo(OUTPUT));
    assertThat(s.getFields().get(1).schema().getType(), equalTo(Schema.Type.RECORD));
    assertThat(s.getFields().get(1).schema().getName(), equalTo("ModelAll_Output"));
    assertThat(s.getFields().get(1).schema().getFields(), hasSize(2));

    assertThat(s.getFields().get(1).schema().getFields().get(0).name(), equalTo("o1"));
    assertThat(
        s.getFields().get(1).schema().getFields().get(0).schema().getType(),
        equalTo(Schema.Type.LONG));
    assertThat(
        s.getFields().get(1).schema().getFields().get(0).getProp(SQL_TYPE), equalTo("INT64"));
    assertThat(
        s.getFields().get(1).schema().getFields().get(0).getProp(SPANNER_OPTION + "0"),
        equalTo("required=TRUE"));
    assertThat(s.getFields().get(1).schema().getFields().get(1).name(), equalTo("o2"));
    assertThat(
        s.getFields().get(1).schema().getFields().get(1).schema().getType(),
        equalTo(Schema.Type.DOUBLE));
    assertThat(
        s.getFields().get(1).schema().getFields().get(1).getProp(SQL_TYPE), equalTo("FLOAT64"));

    s = iterator.next();
    assertThat(s.getName(), equalTo("ModelMin"));
    assertThat(s.getNamespace(), equalTo("spannertest"));
    assertThat(s.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(s.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));
    assertThat(s.getProp(SPANNER_ENTITY), equalTo(SPANNER_ENTITY_MODEL));
    assertThat(s.getProp(SPANNER_REMOTE), equalTo("false"));
    assertThat(s.getFields(), hasSize(2));
    assertThat(s.getFields().get(0).name(), equalTo(INPUT));
    assertThat(s.getFields().get(0).schema().getType(), equalTo(Schema.Type.RECORD));
    assertThat(s.getFields().get(0).schema().getFields(), hasSize(1));
    assertThat(s.getFields().get(1).name(), equalTo(OUTPUT));
    assertThat(s.getFields().get(1).schema().getType(), equalTo(Schema.Type.RECORD));
    assertThat(s.getFields().get(0).schema().getFields(), hasSize(1));

    s = iterator.next();
    assertThat(s.getName(), equalTo("ModelStruct"));
    assertThat(s.getFields().get(0).name(), equalTo(INPUT));
    assertThat(s.getFields().get(0).schema().getType(), equalTo(Schema.Type.RECORD));
    assertThat(s.getFields().get(0).schema().getFields(), hasSize(1));
    assertThat(s.getFields().get(0).schema().getFields().get(0).name(), equalTo("i1"));
    assertThat(
        s.getFields().get(0).schema().getFields().get(0).schema(),
        equalTo(
            SchemaBuilder.builder()
                .record("struct_ModelStruct_input_0")
                .fields()
                .name("a")
                .type()
                .booleanType()
                .noDefault()
                .endRecord()));
    assertThat(s.getFields().get(1).name(), equalTo(OUTPUT));
    assertThat(s.getFields().get(1).schema().getType(), equalTo(Schema.Type.RECORD));
    assertThat(s.getFields().get(1).schema().getFields(), hasSize(1));
    assertThat(s.getFields().get(1).schema().getFields().get(0).name(), equalTo("o1"));
    assertThat(
        s.getFields().get(1).schema().getFields().get(0).schema().toString(),
        equalTo(
            SchemaBuilder.builder() //
                .record("struct_ModelStruct_output_0")
                .fields()
                .name("a")
                .type()
                .booleanType()
                .noDefault() //
                .name("b")
                .type()
                .array()
                .items()
                .unionOf()
                .nullType()
                .and()
                .record("struct_ModelStruct_output_0_1")
                .fields()
                .name("c")
                .type()
                .stringType()
                .noDefault()
                .name("d")
                .type()
                .array()
                .items()
                .unionOf()
                .nullType()
                .and()
                .doubleType()
                .endUnion()
                .noDefault()
                .endRecord()
                .endUnion()
                .noDefault()
                .name("e")
                .type()
                .record("struct_ModelStruct_output_0_2")
                .fields()
                .name("f")
                .type()
                .record("struct_ModelStruct_output_0_2_0")
                .fields()
                .name("g")
                .type()
                .longType()
                .noDefault()
                .endRecord()
                .noDefault()
                .endRecord()
                .noDefault()
                .endRecord()
                .toString()));

    assertThat(iterator.hasNext(), equalTo(false));
  }

  @Test
  public void changeStreams() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", true);
    Ddl ddl =
        Ddl.builder()
            .createChangeStream("ChangeStreamAll")
            .forClause("FOR ALL")
            .options(
                ImmutableList.of(
                    "retention_period=\"7d\"", "value_capture_type=\"OLD_AND_NEW_VALUES\""))
            .endChangeStream()
            .createChangeStream("ChangeStreamEmpty")
            .endChangeStream()
            .createChangeStream("ChangeStreamTableColumns")
            .forClause("FOR `T1`, `T2`(`c1`, `c2`), `T3`()")
            .endChangeStream()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(3));
    for (Schema s : result) {
      assertThat(s.getNamespace(), equalTo("spannertest"));
      assertThat(s.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
      assertThat(s.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));
      assertThat(s.getFields(), empty());
    }

    Iterator<Schema> it = result.iterator();
    Schema avroSchema1 = it.next();
    assertThat(avroSchema1.getName(), equalTo("ChangeStreamAll"));
    assertThat(avroSchema1.getProp(SPANNER_CHANGE_STREAM_FOR_CLAUSE), equalTo("FOR ALL"));
    assertThat(avroSchema1.getProp(SPANNER_OPTION + "0"), equalTo("retention_period=\"7d\""));
    assertThat(
        avroSchema1.getProp(SPANNER_OPTION + "1"),
        equalTo("value_capture_type=\"OLD_AND_NEW_VALUES\""));

    Schema avroSchema2 = it.next();
    assertThat(avroSchema2.getName(), equalTo("ChangeStreamEmpty"));
    assertThat(avroSchema2.getProp(SPANNER_CHANGE_STREAM_FOR_CLAUSE), equalTo(""));
    assertThat(avroSchema2.getProp(SPANNER_OPTION + "0"), nullValue());

    Schema avroSchema3 = it.next();
    assertThat(avroSchema3.getName(), equalTo("ChangeStreamTableColumns"));
    assertThat(
        avroSchema3.getProp(SPANNER_CHANGE_STREAM_FOR_CLAUSE),
        equalTo("FOR `T1`, `T2`(`c1`, `c2`), `T3`()"));
    assertThat(avroSchema3.getProp(SPANNER_OPTION + "0"), nullValue());
  }

  @Test
  public void pgChangeStreams() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", true);
    Ddl ddl =
        Ddl.builder(Dialect.POSTGRESQL)
            .createChangeStream("ChangeStreamAll")
            .forClause("FOR ALL")
            .options(
                ImmutableList.of(
                    "retention_period='7d'", "value_capture_type='OLD_AND_NEW_VALUES'"))
            .endChangeStream()
            .createChangeStream("ChangeStreamEmpty")
            .endChangeStream()
            .createChangeStream("ChangeStreamTableColumns")
            .forClause("FOR \"T1\", \"T2\"(\"c1\", \"c2\"), \"T3\"()")
            .endChangeStream()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(3));
    for (Schema s : result) {
      assertThat(s.getNamespace(), equalTo("spannertest"));
      assertThat(s.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
      assertThat(s.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));
      assertThat(s.getFields(), empty());
    }

    Iterator<Schema> it = result.iterator();
    Schema avroSchema1 = it.next();
    assertThat(avroSchema1.getName(), equalTo("ChangeStreamAll"));
    assertThat(avroSchema1.getProp(SPANNER_CHANGE_STREAM_FOR_CLAUSE), equalTo("FOR ALL"));
    assertThat(avroSchema1.getProp(SPANNER_OPTION + "0"), equalTo("retention_period='7d'"));
    assertThat(
        avroSchema1.getProp(SPANNER_OPTION + "1"),
        equalTo("value_capture_type='OLD_AND_NEW_VALUES'"));

    Schema avroSchema2 = it.next();
    assertThat(avroSchema2.getName(), equalTo("ChangeStreamEmpty"));
    assertThat(avroSchema2.getProp(SPANNER_CHANGE_STREAM_FOR_CLAUSE), equalTo(""));
    assertThat(avroSchema2.getProp(SPANNER_OPTION + "0"), nullValue());

    Schema avroSchema3 = it.next();
    assertThat(avroSchema3.getName(), equalTo("ChangeStreamTableColumns"));
    assertThat(
        avroSchema3.getProp(SPANNER_CHANGE_STREAM_FOR_CLAUSE),
        equalTo("FOR \"T1\", \"T2\"(\"c1\", \"c2\"), \"T3\"()"));
    assertThat(avroSchema3.getProp(SPANNER_OPTION + "0"), nullValue());
  }

  @Test
  public void sequences() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", true);
    Ddl ddl =
        Ddl.builder()
            .createSequence("Sequence1")
            .options(
                ImmutableList.of(
                    "sequence_kind=\"bit_reversed_positive\"",
                    "skip_range_min=0",
                    "skip_range_max=1000",
                    "start_with_counter=50"))
            .endSequence()
            .createSequence("Sequence2")
            .options(
                ImmutableList.of(
                    "sequence_kind=\"bit_reversed_positive\"", "start_with_counter=9999"))
            .endSequence()
            .createSequence("Sequence3")
            .options(ImmutableList.of("sequence_kind=\"bit_reversed_positive\""))
            .endSequence()
            .createSequence("Sequence4")
            .options(ImmutableList.of("sequence_kind=\"default\""))
            .endSequence()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(4));
    for (Schema s : result) {
      assertThat(s.getNamespace(), equalTo("spannertest"));
      assertThat(s.getProp("googleFormatVersion"), equalTo("booleans"));
      assertThat(s.getProp("googleStorage"), equalTo("CloudSpanner"));
      assertThat(s.getFields(), empty());
    }

    Iterator<Schema> it = result.iterator();
    Schema avroSchema1 = it.next();
    assertThat(avroSchema1.getName(), equalTo("Sequence1"));
    assertThat(
        avroSchema1.getProp("sequenceOption_0"),
        equalTo("sequence_kind=\"bit_reversed_positive\""));
    assertThat(avroSchema1.getProp("sequenceOption_1"), equalTo("skip_range_min=0"));
    assertThat(avroSchema1.getProp("sequenceOption_2"), equalTo("skip_range_max=1000"));
    assertThat(avroSchema1.getProp("sequenceOption_3"), equalTo("start_with_counter=50"));

    Schema avroSchema2 = it.next();
    assertThat(avroSchema2.getName(), equalTo("Sequence2"));
    assertThat(
        avroSchema2.getProp("sequenceOption_0"),
        equalTo("sequence_kind=\"bit_reversed_positive\""));
    assertThat(avroSchema2.getProp("sequenceOption_1"), equalTo("start_with_counter=9999"));

    Schema avroSchema3 = it.next();
    assertThat(avroSchema3.getName(), equalTo("Sequence3"));
    assertThat(
        avroSchema3.getProp("sequenceOption_0"),
        equalTo("sequence_kind=\"bit_reversed_positive\""));

    Schema avroSchema4 = it.next();
    assertThat(avroSchema4.getName(), equalTo("Sequence4"));
    assertThat(avroSchema4.getProp("sequenceOption_0"), equalTo("sequence_kind=\"default\""));
  }

  @Test
  public void pgSequences() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", true);
    Ddl ddl =
        Ddl.builder(Dialect.POSTGRESQL)
            .createSequence("PGSequence1")
            .sequenceKind("bit_reversed_positive")
            .counterStartValue(Long.valueOf(50))
            .skipRangeMin(Long.valueOf(0))
            .skipRangeMax(Long.valueOf(1000))
            .endSequence()
            .createSequence("PGSequence2")
            .sequenceKind("bit_reversed_positive")
            .counterStartValue(Long.valueOf(9999))
            .endSequence()
            .createSequence("PGSequence3")
            .sequenceKind("bit_reversed_positive")
            .endSequence()
            .createSequence("PGSequence4")
            .sequenceKind("default")
            .endSequence()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(4));
    for (Schema s : result) {
      assertThat(s.getNamespace(), equalTo("spannertest"));
      assertThat(s.getProp("googleFormatVersion"), equalTo("booleans"));
      assertThat(s.getProp("googleStorage"), equalTo("CloudSpanner"));
      assertThat(s.getFields(), empty());
    }

    Iterator<Schema> it = result.iterator();
    Schema avroSchema1 = it.next();
    assertThat(avroSchema1.getName(), equalTo("PGSequence1"));
    assertThat(avroSchema1.getProp(SPANNER_SEQUENCE_KIND), equalTo("bit_reversed_positive"));
    assertThat(avroSchema1.getProp(SPANNER_SEQUENCE_SKIP_RANGE_MIN), equalTo("0"));
    assertThat(avroSchema1.getProp(SPANNER_SEQUENCE_SKIP_RANGE_MAX), equalTo("1000"));
    assertThat(avroSchema1.getProp(SPANNER_SEQUENCE_COUNTER_START), equalTo("50"));

    Schema avroSchema2 = it.next();
    assertThat(avroSchema2.getName(), equalTo("PGSequence2"));
    assertThat(avroSchema2.getProp(SPANNER_SEQUENCE_KIND), equalTo("bit_reversed_positive"));
    assertThat(avroSchema2.getProp(SPANNER_SEQUENCE_COUNTER_START), equalTo("9999"));

    Schema avroSchema3 = it.next();
    assertThat(avroSchema3.getName(), equalTo("PGSequence3"));
    assertThat(avroSchema3.getProp(SPANNER_SEQUENCE_KIND), equalTo("bit_reversed_positive"));

    Schema avroSchema4 = it.next();
    assertThat(avroSchema4.getName(), equalTo("PGSequence4"));
    assertThat(avroSchema4.getProp(SPANNER_SEQUENCE_KIND), equalTo("default"));
  }

  @Test
  public void placements() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    String placementName1 = "pl1";
    String placementName2 = "pl2";
    String instancePartition = "mr-partition";
    String defaultLeader = "us-central1";
    Ddl ddl =
        Ddl.builder()
            .createPlacement(placementName1)
            .options(ImmutableList.of("instance_partition=\"" + instancePartition + "\""))
            .endPlacement()
            .createPlacement(placementName2)
            .options(
                ImmutableList.of(
                    "instance_partition=\"" + instancePartition + "\"",
                    "default_leader=\"" + defaultLeader + "\""))
            .endPlacement()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(2));

    Iterator<Schema> it = result.iterator();
    Schema avroSchema1 = it.next();
    assertThat(avroSchema1.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema1.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema1.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));
    assertThat(avroSchema1.getProp(SPANNER_ENTITY), equalTo(SPANNER_ENTITY_PLACEMENT));
    assertThat(avroSchema1.getFields(), empty());
    assertThat(avroSchema1.getName(), equalTo(placementName1));
    assertThat(
        avroSchema1.getProp(SPANNER_OPTION + "0"),
        equalTo("instance_partition=\"" + instancePartition + "\""));

    Schema avroSchema2 = it.next();
    assertThat(avroSchema2.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema2.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema2.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));
    assertThat(avroSchema2.getProp(SPANNER_ENTITY), equalTo(SPANNER_ENTITY_PLACEMENT));
    assertThat(avroSchema2.getFields(), empty());
    assertThat(avroSchema2.getName(), equalTo(placementName2));
    assertThat(
        avroSchema2.getProp(SPANNER_OPTION + "0"),
        equalTo("instance_partition=\"" + instancePartition + "\""));
    assertThat(
        avroSchema2.getProp(SPANNER_OPTION + "1"),
        equalTo("default_leader=\"" + defaultLeader + "\""));
  }

  @Test
  public void placementTable() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder()
            .createTable("PlacementKeyWithPrimaryKey")
            .column("location")
            .type(Type.string())
            .max()
            .notNull()
            .placementKey()
            .endColumn()
            .column("val")
            .type(Type.string())
            .size(10)
            .endColumn()
            .endTable()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroSchema = result.iterator().next();

    assertThat(avroSchema.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));

    assertThat(avroSchema.getName(), equalTo("PlacementKeyWithPrimaryKey"));

    List<Schema.Field> fields = avroSchema.getFields();

    assertThat(fields, hasSize(2));

    // Placement key column
    assertThat(fields.get(0).name(), equalTo("location"));
    assertThat(fields.get(0).schema().getType(), equalTo(Schema.Type.STRING));
    assertThat(fields.get(0).getProp(SQL_TYPE), equalTo("STRING(MAX)"));
    assertThat(fields.get(0).getProp(SPANNER_PLACEMENT_KEY), equalTo("true"));
    assertThat(fields.get(0).getProp(STORED), equalTo(null));

    assertThat(fields.get(1).name(), equalTo("val"));
    assertThat(fields.get(1).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(1).getProp(SQL_TYPE), equalTo("STRING(10)"));
    assertThat(fields.get(1).getProp(SPANNER_PLACEMENT_KEY), equalTo(null));
  }

  @Test
  public void pgPlacementTable() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder(Dialect.POSTGRESQL)
            .createTable("PlacementKeyWithPrimaryKey")
            .column("location")
            .pgVarchar()
            .max()
            .notNull()
            .placementKey()
            .endColumn()
            .column("val")
            .pgVarchar()
            .size(10)
            .endColumn()
            .endTable()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroSchema = result.iterator().next();

    assertThat(avroSchema.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));

    assertThat(avroSchema.getName(), equalTo("PlacementKeyWithPrimaryKey"));

    List<Schema.Field> fields = avroSchema.getFields();

    assertThat(fields, hasSize(2));

    // Placement key column
    assertThat(fields.get(0).name(), equalTo("location"));
    assertThat(fields.get(0).schema().getType(), equalTo(Schema.Type.STRING));
    assertThat(fields.get(0).getProp(SQL_TYPE), equalTo("character varying"));
    assertThat(fields.get(0).getProp(SPANNER_PLACEMENT_KEY), equalTo("true"));
    assertThat(fields.get(0).getProp(STORED), equalTo(null));

    assertThat(fields.get(1).name(), equalTo("val"));
    assertThat(fields.get(1).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(1).getProp(SQL_TYPE), equalTo("character varying(10)"));
    assertThat(fields.get(1).getProp(SPANNER_PLACEMENT_KEY), equalTo(null));
  }

  @Test
  public void interleaveInTable() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder()
            .createTable("InterleaveInTable")
            .column("k1")
            .type(Type.string())
            .max()
            .notNull()
            .endColumn()
            .column("v1")
            .type(Type.string())
            .size(10)
            .endColumn()
            .interleaveInParent("ParentTable")
            .interleaveType(InterleaveType.IN)
            .endTable()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroSchema = result.iterator().next();

    assertThat(avroSchema.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));

    assertThat(avroSchema.getName(), equalTo("InterleaveInTable"));

    List<Schema.Field> fields = avroSchema.getFields();

    assertThat(fields, hasSize(2));

    // k1
    assertThat(fields.get(0).name(), equalTo("k1"));
    assertThat(fields.get(0).schema().getType(), equalTo(Schema.Type.STRING));
    assertThat(fields.get(0).getProp(SQL_TYPE), equalTo("STRING(MAX)"));

    // v1
    assertThat(fields.get(1).name(), equalTo("v1"));
    assertThat(fields.get(1).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(1).getProp(SQL_TYPE), equalTo("STRING(10)"));

    assertThat(avroSchema.getProp(SPANNER_PARENT), equalTo("ParentTable"));
    assertThat(avroSchema.getProp(SPANNER_ON_DELETE_ACTION), equalTo("no action"));
    assertThat(avroSchema.getProp(SPANNER_INTERLEAVE_TYPE), equalTo("IN"));
  }

  @Test
  public void pgInterleaveInTable() {
    DdlToAvroSchemaConverter converter =
        new DdlToAvroSchemaConverter("spannertest", "booleans", false);
    Ddl ddl =
        Ddl.builder(Dialect.POSTGRESQL)
            .createTable("InterleaveInTable")
            .column("k1")
            .type(Type.string())
            .max()
            .notNull()
            .endColumn()
            .column("v1")
            .type(Type.string())
            .size(10)
            .endColumn()
            .interleaveInParent("ParentTable")
            .interleaveType(InterleaveType.IN)
            .endTable()
            .build();

    Collection<Schema> result = converter.convert(ddl);
    assertThat(result, hasSize(1));
    Schema avroSchema = result.iterator().next();

    assertThat(avroSchema.getNamespace(), equalTo("spannertest"));
    assertThat(avroSchema.getProp(GOOGLE_FORMAT_VERSION), equalTo("booleans"));
    assertThat(avroSchema.getProp(GOOGLE_STORAGE), equalTo("CloudSpanner"));

    assertThat(avroSchema.getName(), equalTo("InterleaveInTable"));

    List<Schema.Field> fields = avroSchema.getFields();

    assertThat(fields, hasSize(2));

    // k1
    assertThat(fields.get(0).name(), equalTo("k1"));
    assertThat(fields.get(0).schema().getType(), equalTo(Schema.Type.STRING));
    assertThat(fields.get(0).getProp(SQL_TYPE), equalTo("STRING(MAX)"));

    // v1
    assertThat(fields.get(1).name(), equalTo("v1"));
    assertThat(fields.get(1).schema(), equalTo(nullableUnion(Schema.Type.STRING)));
    assertThat(fields.get(1).getProp(SQL_TYPE), equalTo("STRING(10)"));

    assertThat(avroSchema.getProp(SPANNER_PARENT), equalTo("ParentTable"));
    assertThat(avroSchema.getProp(SPANNER_ON_DELETE_ACTION), equalTo("no action"));
    assertThat(avroSchema.getProp(SPANNER_INTERLEAVE_TYPE), equalTo("IN"));
  }

  private Schema nullableUnion(Schema.Type s) {
    return Schema.createUnion(Schema.create(Schema.Type.NULL), Schema.create(s));
  }

  private Schema nullableUuid() {
    return Schema.createUnion(
        Schema.create(Schema.Type.NULL),
        LogicalTypes.uuid().addToSchema(Schema.create(Schema.Type.STRING)));
  }

  private Schema nullableNumericUnion() {
    return Schema.createUnion(
        Schema.create(Schema.Type.NULL),
        LogicalTypes.decimal(NumericUtils.PRECISION, NumericUtils.SCALE)
            .addToSchema(Schema.create(Schema.Type.BYTES)));
  }

  private Schema nullableTimestampUnion() {
    return Schema.createUnion(
        Schema.create(Schema.Type.NULL),
        LogicalTypes.timestampMicros().addToSchema(Schema.create(Schema.Type.LONG)));
  }

  private Schema nullableArray(Schema.Type s) {
    return Schema.createUnion(
        Schema.create(Schema.Type.NULL),
        Schema.createArray(Schema.createUnion(Schema.create(Schema.Type.NULL), Schema.create(s))));
  }

  private Schema nullableUuidArray() {
    return Schema.createUnion(
        Schema.create(Schema.Type.NULL),
        Schema.createArray(
            Schema.createUnion(
                Schema.create(Schema.Type.NULL),
                LogicalTypes.uuid().addToSchema(Schema.create(Schema.Type.STRING)))));
  }

  private Schema nullableNumericArray() {
    return Schema.createUnion(
        Schema.create(Schema.Type.NULL),
        Schema.createArray(
            Schema.createUnion(
                Schema.create(Schema.Type.NULL),
                LogicalTypes.decimal(NumericUtils.PRECISION, NumericUtils.SCALE)
                    .addToSchema(Schema.create(Schema.Type.BYTES)))));
  }

  private Schema nullablePgNumericArray() {
    return Schema.createUnion(
        Schema.create(Schema.Type.NULL),
        Schema.createArray(
            Schema.createUnion(
                Schema.create(Schema.Type.NULL),
                LogicalTypes.decimal(NumericUtils.PG_MAX_PRECISION, NumericUtils.PG_MAX_SCALE)
                    .addToSchema(Schema.create(Schema.Type.BYTES)))));
  }

  private Schema nullablePgNumericUnion() {
    return Schema.createUnion(
        Schema.create(Schema.Type.NULL),
        LogicalTypes.decimal(NumericUtils.PG_MAX_PRECISION, NumericUtils.PG_MAX_SCALE)
            .addToSchema(Schema.create(Schema.Type.BYTES)));
  }
}
